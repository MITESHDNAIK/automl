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

from ml_utils import preprocess, detect_task
from pandas.api.types import is_numeric_dtype

app = FastAPI(title="AutoML API", version="1.0.0")

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_REGISTRY = {
    "classification": {
        "Decision Tree": lambda p: DecisionTreeClassifier(max_depth=p.get("max_depth"), random_state=42),
        "Random Forest": lambda p: RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth"), random_state=42),
        "Logistic Regression": lambda p: LogisticRegression(max_iter=1000, random_state=42),
        "SVM": lambda p: SVC(kernel=p.get("kernel", "rbf"), random_state=42),
        "KNN": lambda p: KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5)),
        "Naive Bayes": lambda p: GaussianNB()
    },
    "regression": {
        "Linear Regression": lambda p: LinearRegression(),
        "Decision Tree": lambda p: DecisionTreeRegressor(max_depth=p.get("max_depth"), random_state=42),
        "Random Forest": lambda p: RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth"), random_state=42),
        "SVR": lambda p: SVR(kernel=p.get("kernel", "rbf")),
        "KNN": lambda p: KNeighborsRegressor(n_neighbors=p.get("n_neighbors", 5))
    },
    "unsupervised": {
        "KMeans": lambda p: KMeans(n_clusters=p.get("n_clusters", 3), random_state=42, n_init=10),
        "PCA": lambda p: PCA(n_components=p.get("n_components", 2))
    }
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": "Best for linear relationships. Interpretable coefficients.",
    "Logistic Regression": "Excellent for classification. Provides probability estimates.",
    "Decision Tree": "Highly interpretable, handles mixed data. Prone to overfitting.",
    "Random Forest": "Robust ensemble, reduces overfitting. Good with missing values.",
    "SVM": "Powerful for high-dimensional data and complex boundaries.",
    "KNN": "Simple algorithm for small datasets with clear local patterns.",
    "Naive Bayes": "Fast probabilistic classifier assuming feature independence.",
    "KMeans": "Unsupervised clustering for finding natural groupings in data.",
    "PCA": "Dimensionality reduction technique preserving most variance.",
    "SVR": "Support Vector Regression for non-linear regression problems."
}

MODEL_DESC_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k, v in MODEL_DESCRIPTIONS.items()}

@app.get("/")
def read_root():
    return {"message": "AutoML API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Helper function to get all column data for plotting
def get_all_column_data_for_plotting(df: pd.DataFrame):
    if len(df) > 2000:
        df_sample = df.sample(n=2000, random_state=42)
    else:
        df_sample = df
    plots_data = {}
    for col in df_sample.columns:
        cleaned_series = df_sample[col].where(pd.notna(df_sample[col]), None)
        plots_data[col] = cleaned_series.tolist()
    return plots_data

def create_feature_importance_plot(model, feature_names, model_name):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return None
        indices = np.argsort(importances)[::-1][:10]
        fig = go.Figure([go.Bar(x=importances[indices], y=[feature_names[i] for i in indices], orientation='h')])
        fig.update_layout(title=f'Top 10 Feature Importance - {model_name}', yaxis={'autorange': 'reversed'})
        return fig.to_json()
    except Exception as e:
        return None

def entropy_id3_gain(y: pd.Series, x: pd.Series) -> float:
    from scipy.stats import entropy
    y, x = y.astype(str), x.astype(str)
    Hy = entropy(y.value_counts(normalize=True), base=2)
    Hy_x = sum(p_x * entropy(grp.value_counts(normalize=True), base=2) for val, grp in y.groupby(x) if (p_x := len(grp) / len(y)) > 0)
    return Hy - Hy_x

class EntropyRequest(BaseModel):
    upload_path: str
    target_column: str

@app.post("/entropy_gain")
def entropy_gain(req: EntropyRequest):
    try:
        df = pd.read_csv(req.upload_path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(req.upload_path, sep=";", engine="python", on_bad_lines="skip")
    target = req.target_column if req.target_column in df.columns else df.columns[-1]
    gains = {col: entropy_id3_gain(df[target], df[col]) for col in df.select_dtypes(include=["object", "category", "bool"]) if col != target}
    sorted_gain = dict(sorted(gains.items(), key=lambda item: item[1], reverse=True))
    return {"columns": list(sorted_gain.keys()), "gains": list(sorted_gain.values())}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), engine='python', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(io.BytesIO(contents), sep=';', engine='python', on_bad_lines='skip')
    target = df.columns[-1]
    stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(str).to_dict(),
        "n_missing": df.isnull().sum().to_dict(),
        "target": target,
    }
    data_for_plotting = get_all_column_data_for_plotting(df)
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)
    return {
        "upload_path": save_path,
        "stats": stats,
        "data_for_plotting": data_for_plotting
    }

class TrainRequest(BaseModel):
    upload_path: str
    target_column: str
    test_size: float = 0.2
    max_depth: int = None
    n_estimators: int = 100
    kernel: str = "rbf"
    n_neighbors: int = 5

@app.post("/train")
def train(req: TrainRequest):
    df = pd.read_csv(req.upload_path, engine='python', on_bad_lines='skip')
    
    # 1. Get the original, un-processed target column for final metric calculation (MSE)
    # FIX: Must reset index to align with X after preprocessing in ml_utils.py
    y_original_full = df[req.target_column].copy().reset_index(drop=True)
    
    # 2. MODIFIED UNPACKING: now receiving 6 values (X, y_processed, X_scaled, target_encoder, feature_scaler, y_scaler)
    X, y_processed, X_scaled, target_encoder, feature_scaler, y_scaler = preprocess(df, req.target_column)
    
    task = detect_task(y_processed)
    results = {}
    best_score = -float('inf')
    best_model_name = None
    best_model = None

    if task in ["classification", "regression"]:
        # ----------  SAFE STRATIFY LOGIC  ----------
        stratify = None
        if task == "classification":
          n_samples = len(y_processed)
          n_classes = pd.Series(y_processed).nunique()
        # sklearn demands: test_size >= n_classes  when stratify is used
          if n_samples >= 5 and req.test_size * n_samples >= n_classes:
            min_class_size = pd.Series(y_processed).value_counts().min()
            stratify = y_processed if min_class_size >= 2 else None
        
        # 3. Split all data consistently
        X_train, X_test, y_train_processed, y_test_processed = train_test_split(
            X, y_processed, test_size=req.test_size, random_state=42, stratify=stratify
        )
        # Split scaled features with the processed target
        X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y_processed, test_size=req.test_size, random_state=42, stratify=stratify)
        
        # 4. Split the original target for final regression metric calculation (MSE)
        # y_original_full is now aligned because of the index reset above (Line 185)
        _, _, y_test_original, _ = train_test_split(
            X, y_original_full, test_size=req.test_size, random_state=42, stratify=None 
        )
        # ----------  END SAFE STRATIFY  ----------
        
        for name, builder in MODEL_REGISTRY[task].items():
            try:
                use_scaled = name in ["SVM", "SVR", "KNN", "Logistic Regression", "Linear Regression"]
                X_tr, X_te = (X_train_scaled, X_test_scaled) if use_scaled else (X_train, X_test)
                
                model = builder(req.dict())
                model.fit(X_tr, y_train_processed)
                pred_processed = model.predict(X_te) # Prediction is in processed space (encoded or scaled)

                if task == "classification":
                    # Classification metrics: use processed (encoded) prediction and test set
                    score = f1_score(y_test_processed, pred_processed, average="macro", zero_division=0)
                    results[name] = {"accuracy": accuracy_score(y_test_processed, pred_processed), "f1_macro": score, "score": score}
                else: # Regression
                    # FIX: Inverse transform predictions for meaningful MSE calculation
                    pred = y_scaler.inverse_transform(pred_processed.reshape(-1, 1)).flatten()
                    
                    r2_val = r2_score(y_test_processed, pred_processed) # R2 is scale-invariant, use scaled data for consistency with model training
                    mse_val = mean_squared_error(y_test_original, pred) # MSE uses original unscaled data

                    score = r2_val
                    results[name] = {"r2": r2_val, "mse": mse_val, "score": score}

                if score > best_score:
                    best_score, best_model_name, best_model = score, name, model
            except Exception as e:
                # Store the error message
                results[name] = {"error": str(e), "score": -1}

        # --- Start of Safety Block to prevent AttributeError: 'NoneType' object has no attribute 'predict' ---
        
        df_stats = {"rows": df.shape[0], "cols": df.shape[1], "task": task}
        context = f"Dataset: {df_stats['rows']} rows, {df_stats['cols']} features. Task: {task}."
        ctx_emb = embedder.encode(context, convert_to_tensor=True)
        sims = {k: float(util.cos_sim(ctx_emb, MODEL_DESC_EMB[k])) for k in MODEL_DESCRIPTIONS}

        # Initialize dependent variables to safe defaults
        explanation = "Multiple algorithms were tested, but an error prevented selection of a best model or successful training of any model. Review the error column for details."
        confusion_plotly_data = None
        feature_importance_plotly_data = None
        
        # Only proceed with prediction and detailed analysis if a best model was found
        if best_model:
            performance_text = f"achieving a score of {best_score:.3f}."
            explanation = f"Based on your dataset, {best_model_name} performed best, {performance_text} {MODEL_DESCRIPTIONS.get(best_model_name, '')}"

            # Get best model prediction for confusion matrix (must use processed data)
            X_test_best = X_test_scaled if best_model_name in ["SVM", "SVR", "KNN", "Logistic Regression", "Linear Regression"] else X_test
            y_pred_best = best_model.predict(X_test_best) 
            
            if task == "classification":
                confusion_plotly_data = json.dumps({
                    "data": [{"z": confusion_matrix(y_test_processed, y_pred_best).tolist(), "type": "heatmap", "colorscale": "Blues"}],
                    "layout": {"title": f"Confusion Matrix – {best_model_name}", "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Actual"}}
                })
            
            feature_importance_plotly_data = create_feature_importance_plot(best_model, X.columns.tolist(), best_model_name)

        # --- End of Safety Block ---

        return {
            "task": task,
            "results": results,
            "best_model": best_model_name,
            "perf_plotly": json.dumps({
                "data": [
                    {
                        "x": list(results.keys()),
                        "y": [r.get("accuracy" if task == "classification" else "r2", 0) for r in results.values()],
                        "type": "bar",
                        "name": "Accuracy" if task == "classification" else "R²",
                        "marker": {"color": "#3B82F6"}
                    }
                ],
                "layout": {"title": "Model Comparison", "xaxis": {"title": "Model"}, "yaxis": {"title": "Score", "range": [0, 1] if task == 'classification' else [-1, 1]}}
            }),
            "confusion_plotly": confusion_plotly_data,
            "feature_importance_plotly": feature_importance_plotly_data,
            "explanation": explanation,
            "dataset_stats": df_stats
        }
    
    # Return generic error for unsupported tasks (e.g., Unsupervised)
    return {"error": "Unsupervised task not fully implemented"}