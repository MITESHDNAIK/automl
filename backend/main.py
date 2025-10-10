# backend/main.py
import os, io, joblib, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util

# Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# REMOVED: XGBoost imports

from ml_utils import load_df, preprocess, detect_task
from pandas.api.types import is_numeric_dtype

app = FastAPI(title="AutoML API", version="1.0.0")

# Enhanced CORS configuration (Confirmed to be correct)
app.add_middleware(
    CORSMiddleware,
    # Added explicit localhost and 127.0.0.1 origins for common ports
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000", "http://localhost:8000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load small embedding model (CPU friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Registry of algorithms with proper parameter handling
MODEL_REGISTRY = {
    "classification": {
        "Decision Tree": lambda p: DecisionTreeClassifier(
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42
        ),
        "Random Forest": lambda p: RandomForestClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42
        ),
        "Logistic Regression": lambda p: LogisticRegression(
            max_iter=1000, 
            random_state=42
        ),
        "SVM": lambda p: SVC(
            kernel=p.get("kernel", "rbf"),
            random_state=42
        ),
        "KNN": lambda p: KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 5)
        ),
        "Naive Bayes": lambda p: GaussianNB()
    },
    "regression": {
        "Linear Regression": lambda p: LinearRegression(),
        "Decision Tree": lambda p: DecisionTreeRegressor(
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42
        ),
        "Random Forest": lambda p: RandomForestRegressor(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42
        ),
        "SVR": lambda p: SVR(
            kernel=p.get("kernel", "rbf")
        ),
        "KNN": lambda p: KNeighborsRegressor(
            n_neighbors=p.get("n_neighbors", 5)
        )
    },
    "unsupervised": {
        "KMeans": lambda p: KMeans(
            n_clusters=p.get("n_clusters", 3),
            random_state=42,
            n_init=10
        ),
        "PCA": lambda p: PCA(
            n_components=p.get("n_components", 2)
        )
    }
}

# Enhanced model explanations with dataset characteristics
MODEL_DESCRIPTIONS = {
    "Linear Regression": "Best for linear relationships between features and target. Works well with continuous numerical data and provides interpretable coefficients.",
    "Logistic Regression": "Excellent for binary and multiclass classification. Provides probability estimates and is highly interpretable with feature importance.",
    "Decision Tree": "Highly interpretable model that handles both numerical and categorical data. Good for understanding decision rules but prone to overfitting.",
    "Random Forest": "Robust ensemble method that reduces overfitting. Handles mixed data types well, provides feature importance, and works with missing values.",
    "SVM": "Powerful for high-dimensional data and complex boundaries. Works well with small to medium datasets and can handle non-linear patterns with kernels.",
    "KNN": "Simple algorithm good for small datasets with clear local patterns. No assumptions about data distribution but sensitive to curse of dimensionality.",
    "Naive Bayes": "Fast probabilistic classifier assuming feature independence. Excellent for text classification and high-dimensional sparse data.",
    "KMeans": "Unsupervised clustering for finding natural groupings in data. Best when clusters are spherical and similar in size.",
    "PCA": "Dimensionality reduction technique preserving most variance. Useful for visualization and removing correlated features.",
    "SVR": "Support Vector Regression for non-linear regression problems. Good for high-dimensional data with complex patterns."
}

MODEL_DESC_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k,v in MODEL_DESCRIPTIONS.items()}

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "AutoML API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "algorithms_available": len(MODEL_REGISTRY)}

def create_feature_importance_plot(model, feature_names, model_name):
    """Create feature importance plot for models that support it"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return None
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig = go.Figure([
            go.Bar(
                x=top_importances,
                y=top_features,
                orientation='h',
                marker_color='teal'
            )
        ])
        fig.update_layout(
            title=f'Top 10 Feature Importance - {model_name}',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=400
        )
        return fig.to_json()
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")
        return None

def create_numerical_distribution_plots(df, target_col):
    """Create distribution plots for numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Limit to first 5 numerical columns to avoid too many plots
    numerical_cols = numerical_cols[:5]
    
    plots_data = {}
    for col in numerical_cols:
        if not df[col].empty:
            plots_data[col] = df[col].dropna().tolist()
    
    return plots_data
# ----------  ID3 / Entropy helper  ----------
def entropy_id3_gain(y: pd.Series, x: pd.Series) -> float:
    """
    Classic ID3 information gain:
        Gain = H(y) – H(y|x)
    Both series are treated as categorical.
    """
    from scipy.stats import entropy
    y, x = y.astype(str), x.astype(str)

    # base entropy
    Hy = entropy(y.value_counts(normalize=True), base=2)

    # weighted conditional entropy
    Hy_x = 0.0
    for val, grp in y.groupby(x):
        p_x = len(grp) / len(y)
        Hy_x += p_x * entropy(grp.value_counts(normalize=True), base=2)

    return Hy - Hy_x


# ----------  NEW ENDPOINT  ----------
class EntropyRequest(BaseModel):
    upload_path: str
    target_column: str


@app.post("/entropy_gain")
def entropy_gain(req: EntropyRequest):
    # same robust read you already use
    try:
        df = pd.read_csv(req.upload_path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(req.upload_path, sep=";", engine="python", on_bad_lines="skip")

    target = req.target_column
    if target not in df.columns:
        target = df.columns[-1]

    # compute gain only for categorical-like columns
    gains = {}
    for col in df.select_dtypes(include=["object", "category", "bool"]):
        if col == target:
            continue
        gains[col] = entropy_id3_gain(df[target], df[col])

    # sort high → low
    sorted_gain = dict(sorted(gains.items(), key=lambda kv: kv[1], reverse=True))
    return {
        "columns": list(sorted_gain.keys()),
        "gains":  list(sorted_gain.values()),
    }
# --- Upload Endpoint ---
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), target_column: str = Form(None)):
    contents = await file.read()
    
    # FIX: Use 'python' engine and 'on_bad_lines=skip' for robust CSV reading
    try:
        df = pd.read_csv(io.BytesIO(contents), engine='python', on_bad_lines='skip')
    except Exception as e:
        # Fallback to a common alternative delimiter (e.g., semicolon)
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=';', engine='python', on_bad_lines='skip')
        except Exception:
            # If CSV reading still fails, raise an error
            raise Exception(f"CSV parsing failed: {e}. Please ensure file is correctly formatted or try a different separator.")


    target = target_column if target_column and target_column in df.columns else df.columns[-1]
    preview = df.head(5).to_dict(orient="records")
    stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(str).to_dict(),
        "n_missing": df.isnull().sum().to_dict(),
        "target": target,
    }
    
    # Create numerical distribution data for plots
    numerical_data_for_plot = create_numerical_distribution_plots(df, target)
    
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)

    return {
        "ok": True, 
        "preview": preview, 
        "stats": stats, 
        "upload_path": save_path,
        "numerical_data_for_plot": numerical_data_for_plot
    }

# --- Train Endpoint ---
class TrainRequest(BaseModel):
    upload_path: str
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int = None
    n_estimators: int = 100
    kernel: str = "rbf"
    n_neighbors: int = 5
    n_clusters: int = 3
    n_components: int = 2
    target_column: str = None

@app.post("/train")
def train(req: TrainRequest):
    # FIX: Apply the same robust reading logic as in /upload
    try:
        df = pd.read_csv(req.upload_path, engine='python', on_bad_lines='skip')
    except Exception as e:
        try:
            df = pd.read_csv(req.upload_path, sep=';', engine='python', on_bad_lines='skip')
        except Exception:
            raise Exception("CSV parsing failed in train endpoint. Please ensure file is correctly formatted.")
            
    target_col = req.target_column if req.target_column and req.target_column in df.columns else df.columns[-1]
    
    # FIX: Correct the unpacking to match the 5 values returned by preprocess
    X, y, X_scaled, target_encoder, scaler = preprocess(df, target_col)

    task = detect_task(y)
    print(f"Detected task: {task}")

    results = {}
    best_score = -float('inf')
    best_model_name = None
    best_model = None
    best_pred = None
    y_test_best = None

    if task in ["classification", "regression"]:
        # Split both unscaled (X) and scaled (X_scaled) data consistently
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state, stratify=y if task == "classification" else None)
            
        X_train_scaled, X_test_scaled, _, _ = train_test_split(
            X_scaled, y, test_size=req.test_size, random_state=req.random_state, stratify=y if task == "classification" else None)
            
        for name, builder in MODEL_REGISTRY[task].items():
            try:
                print(f"Training {name}...")
                
                # Determine which data split to use: scaled data for scaling-sensitive models
                if name in ["SVM", "SVR", "KNN"]:
                    X_tr, X_te = X_train_scaled, X_test_scaled
                else:
                    X_tr, X_te = X_train, X_test
                
                model = builder(req.dict())
                model.fit(X_tr, y_train)
                pred = model.predict(X_te)

                if task == "classification":
                    acc = accuracy_score(y_test, pred)
                    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                    score = f1  # Use F1 as primary metric
                    results[name] = {"accuracy": acc, "f1_macro": f1, "score": score}
                else:  # regression
                    r2 = r2_score(y_test, pred)
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mse)
                    score = r2  # Use R2 as primary metric
                    results[name] = {"r2": r2, "mse": mse, "rmse": rmse, "score": score}

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
                    best_pred = pred
                    y_test_best = y_test
                    
                print(f"{name} completed. Score: {score:.4f}")
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {"error": str(e), "score": -1}

        # Save best model
        if best_model:
            model_path = os.path.join(MODEL_DIR, f"{best_model_name.replace(' ', '_')}.joblib")
            joblib.dump(best_model, model_path)
            print(f"Best model saved: {best_model_name} with score {best_score:.4f}")

        # Create performance plots
        perf_plotly = None
        confusion_plotly = None
        feature_importance_plotly = None

        if task == "classification":
            # Performance comparison
            model_names = [name for name, result in results.items() if "error" not in result]
            accuracies = [results[name]["accuracy"] for name in model_names]
            f1_scores = [results[name]["f1_macro"] for name in model_names]
            
            fig = go.Figure([
                go.Bar(name="Accuracy", x=model_names, y=accuracies, marker_color='#3B82F6'),
                go.Bar(name="F1 Macro", x=model_names, y=f1_scores, marker_color='#8B5CF6')
            ])
            fig.update_layout(
                barmode="group", 
                title="Classification Model Performance",
                xaxis_title="Models",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1])
            )
            perf_plotly = fig.to_json()

            # Confusion matrix for best model
            if best_pred is not None and y_test_best is not None:
                # NOTE: The warning "A single label was found..." is due to data imbalance, but the
                # calculation is still performed.
                cm = confusion_matrix(y_test_best, best_pred)
                
                # FIX: Remove invalid 'hoverangles' property that caused the ValueError
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    colorscale='Blues',
                    showscale=True,
                    # Removed: hoverangles="<0.5"
                ))
                fig_cm.update_layout(
                    title=f"Confusion Matrix - {best_model_name}",
                    xaxis_title="Predicted Class",
                    yaxis_title="True Class"
                )
                confusion_plotly = fig_cm.to_json()

        else:  # regression
            model_names = [name for name, result in results.items() if "error" not in result]
            r2_scores = [results[name]["r2"] for name in model_names]
            rmse_scores = [results[name]["rmse"] for name in model_names]
            
            fig = go.Figure([
                go.Bar(name="R²", x=model_names, y=r2_scores, marker_color='#10B981'),
                go.Bar(name="RMSE", x=model_names, y=rmse_scores, marker_color='#F59E0B', yaxis='y2')
            ])
            fig.update_layout(
                title="Regression Model Performance",
                xaxis_title="Models",
                yaxis=dict(title="R² Score", side='left'),
                yaxis2=dict(title="RMSE", side='right', overlaying='y'),
                barmode="group"
            )
            perf_plotly = fig.to_json()

        # Feature importance for best model
        if best_model and hasattr(X, 'columns'):
            feature_importance_plotly = create_feature_importance_plot(
                best_model, X.columns.tolist(), best_model_name
            )

    else:  # unsupervised
        # Use unscaled features for unsupervised models
        for name, builder in MODEL_REGISTRY["unsupervised"].items():
            try:
                model = builder(req.dict())
                model.fit(X)
                if name == "KMeans":
                    inertia = model.inertia_
                    results[name] = {"inertia": inertia}
                elif name == "PCA":
                    var = model.explained_variance_ratio_.tolist()
                    results[name] = {"explained_variance": var}
            except Exception as e:
                results[name] = {"error": str(e)}

    # Enhanced AI recommendation
    df_stats = {
        "rows": df.shape[0],
        "cols": df.shape[1], 
        "task": task,
        "target_unique": len(df[target_col].unique()) if target_col in df.columns else 0,
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }
    
    # Create context for recommendation
    context = f"Dataset: {df_stats['rows']} rows, {df_stats['cols']} features. Task: {task}. "
    if task == "classification":
        context += f"Classes: {df_stats['target_unique']}. "
    context += f"Missing data: {df_stats['missing_pct']:.1f}%."
    
    # Find best matching algorithm description
    ctx_emb = embedder.encode(context, convert_to_tensor=True)
    # Check similarity against all available models
    sims = {k: float(util.cos_sim(ctx_emb, MODEL_DESC_EMB[k])) for k in MODEL_DESCRIPTIONS if k in MODEL_DESCRIPTIONS}
    
    if best_model_name and best_model_name in MODEL_DESCRIPTIONS:
        base_explanation = MODEL_DESCRIPTIONS[best_model_name]
        performance_text = ""
        if task == "classification":
            acc = results[best_model_name]["accuracy"]
            f1 = results[best_model_name]["f1_macro"]
            performance_text = f"achieving {acc:.1%} accuracy and {f1:.3f} F1-macro score"
        elif task == "regression":
            r2 = results[best_model_name]["r2"]
            performance_text = f"achieving an R² score of {r2:.3f}"
            
        explanation = f"Based on your dataset analysis, {best_model_name} performed best, {performance_text}. {base_explanation}"
    else:
        explanation = "Multiple algorithms were tested on your dataset. Check the performance comparison to see detailed results."

    return {
        "ok": True,
        "task": task,
        "results": results,
        "best_model": best_model_name,
        "perf_plotly": perf_plotly,
        "confusion_plotly": confusion_plotly,
        "feature_importance_plotly": feature_importance_plotly,
        "explanation": explanation,
        "dataset_stats": df_stats
    }