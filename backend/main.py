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

MODEL_DESC_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k,v in MODEL_DESCRIPTIONS.items()}

@app.get("/")
def read_root():
    return {"message": "AutoML API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# NEW: Helper function to get all column data for plotting
def get_all_column_data_for_plotting(df: pd.DataFrame):
    """
    Prepares all columns' data for frontend plotting.
    Downsamples large datasets and handles JSON compliance.
    """
    # Downsample if the dataframe is too large to prevent sending huge JSON payloads
    if len(df) > 2000:
        df_sample = df.sample(n=2000, random_state=42)
    else:
        df_sample = df

    plots_data = {}
    for col in df_sample.columns:
        # Replace NaN/NaT with None for JSON compatibility. This fixes the error you saw.
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
    
    # MODIFIED: Use the new function to get data for all columns
    data_for_plotting = get_all_column_data_for_plotting(df)
    
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)

    return {
        "upload_path": save_path,
        "stats": stats,
        "data_for_plotting": data_for_plotting # MODIFIED: New data key for the frontend
    }

class TrainRequest(BaseModel):
    upload_path: str
    target_column: str
    test_size: float = 0.2
    # Add other model params here as needed
    max_depth: int = None
    n_estimators: int = 100
    kernel: str = "rbf"
    n_neighbors: int = 5

@app.post("/train")
def train(req: TrainRequest):
    df = pd.read_csv(req.upload_path, engine='python', on_bad_lines='skip')
    X, y, X_scaled, _, _ = preprocess(df, req.target_column)
    task = detect_task(y)

    results = {}
    best_score = -float('inf')
    best_model_name = None
    best_model = None

    if task in ["classification", "regression"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42, stratify=y if task == "classification" else None)
        X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=req.test_size, random_state=42, stratify=y if task == "classification" else None)

        for name, builder in MODEL_REGISTRY[task].items():
            try:
                use_scaled = name in ["SVM", "SVR", "KNN", "Logistic Regression", "Linear Regression"]
                X_tr, X_te = (X_train_scaled, X_test_scaled) if use_scaled else (X_train, X_test)
                
                model = builder(req.dict())
                model.fit(X_tr, y_train)
                pred = model.predict(X_te)

                if task == "classification":
                    score = f1_score(y_test, pred, average="macro", zero_division=0)
                    results[name] = {"accuracy": accuracy_score(y_test, pred), "f1_macro": score, "score": score}
                else:
                    score = r2_score(y_test, pred)
                    results[name] = {"r2": score, "mse": mean_squared_error(y_test, pred), "score": score}

                if score > best_score:
                    best_score, best_model_name, best_model = score, name, model
            except Exception as e:
                results[name] = {"error": str(e), "score": -1}

        # Post-training analysis
        df_stats = {"rows": df.shape[0], "cols": df.shape[1], "task": task}
        context = f"Dataset: {df_stats['rows']} rows, {df_stats['cols']} features. Task: {task}."
        ctx_emb = embedder.encode(context, convert_to_tensor=True)
        sims = {k: float(util.cos_sim(ctx_emb, MODEL_DESC_EMB[k])) for k in MODEL_DESCRIPTIONS}
        
        explanation = "Multiple algorithms were tested."
        if best_model_name:
            performance_text = f"achieving a score of {best_score:.3f}."
            explanation = f"Based on your dataset, {best_model_name} performed best, {performance_text} {MODEL_DESCRIPTIONS.get(best_model_name, '')}"

        return {
            "task": task,
            "results": results,
            "best_model": best_model_name,
            "feature_importance_plotly": create_feature_importance_plot(best_model, X.columns.tolist(), best_model_name) if best_model else None,
            "explanation": explanation,
        }
    return {"error": "Unsupervised task not fully implemented"}
