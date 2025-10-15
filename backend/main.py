import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
import logging
from .models import MODEL_REGISTRY, MODEL_DESCRIPTIONS
from .utils import init_db, load_df, preprocess, detect_task, id3_gain, create_feature_importance_plot
from .plots import create_performance_plot, create_confusion_matrix
import io
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoML API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)

DB_FILE = "automl.db"
init_db(DB_FILE)

@app.get("/")
def read_root():
    return {"message": "AutoML API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class EntropyRequest(BaseModel):
    upload_path: str
    target_column: str

@app.post("/entropy_gain")
def calc_entropy_gains(req: EntropyRequest):
    df = load_df(req.upload_path)
    target = req.target_column
    categoricals = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    gains = [id3_gain(df, col, target) for col in categoricals]
    return {"columns": categoricals, "gains": gains}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    content = await file.read()
    # Use chunks for large files (>10MB)
    if len(content) > 10 * 1024 * 1024:
        chunks = pd.read_csv(io.BytesIO(content), chunksize=1000)
        df = pd.concat(chunk for chunk in chunks)
    else:
        df = pd.read_csv(io.BytesIO(content))
    upload_path = os.path.join("uploads", file.filename)
    df.to_csv(upload_path, index=False)

    if not target_column:
        target_column = df.columns[-1]  # Default to last column if not specified
        logger.info(f"No target column specified. Defaulting to {target_column}")

    # Validate target column exists
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")

    stats = {
        "shape": list(df.shape),
        "target": target_column,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "n_missing": df.isnull().sum().to_dict()
    }
    numerical_cols = df.select_dtypes(include=np.number).columns.drop(target_column, errors='ignore')
    numerical_data_for_plot = {col: df[col].dropna().tolist() for col in numerical_cols}

    return {
        "upload_path": upload_path,
        "stats": stats,
        "numerical_data_for_plot": numerical_data_for_plot
    }

class TrainRequest(BaseModel):
    upload_path: str
    target_column: str  # Made required to enforce selection
    max_depth: Optional[int] = None
    n_estimators: int = 100
    kernel: str = "rbf"
    n_neighbors: int = 5
    n_clusters: int = 3
    n_components: int = 2

@app.post("/train")
def train_in_background():
    logger.info(f"Starting training for {req.upload_path} with target {req.target_column} at {pd.Timestamp.now()}")
    df = load_df(req.upload_path)
    
    # Validate target column exists
    if req.target_column not in df.columns:
        logger.error(f"Target column '{req.target_column}' not found in dataset")
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found in dataset")

    # Drop rows where target column has NaN
    df = df.dropna(subset=[req.target_column])
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data after dropping NaN values in target column")

    task = detect_task(df)
    
    X, y = prepare_data(df, req.target_column)
    logger.info(f"Task: {task}, X shape: {X.shape}, y shape: {y.shape}")
    
    results = {}
    best_score = -np.inf
    best_model_name = None
    best_model = None
    best_pred = None
    y_test_best = None

    if task in ["classification", "regression"]:
        class_counts = np.bincount(y) if task == "classification" else np.ones(len(y), dtype=int)
        min_samples_per_class = np.min(class_counts[class_counts > 0]) if task == "classification" else len(y)
        total_samples = len(y)

        if total_samples < 20 or (task == "classification" and len(np.unique(y)) / total_samples > 0.5):
            logger.info("Dataset too small for train-test split. Using cross-validation or LOOCV.")
            if min_samples_per_class == 1 and task == "classification":
                logger.info("Using Leave-One-Out Cross-Validation.")
                cv = LeaveOneOut()
            else:
                n_splits = min(5, min_samples_per_class, total_samples)
                logger.info(f"Using {n_splits}-fold cross-validation.")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    model = builder(req.dict())
                    scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]
                        if name in ("SVM", "SVR", "KNN"):
                            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
                        model.fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        if task == "classification":
                            scores.append(f1_score(y_te, pred, average="macro", zero_division=0))
                        else:
                            scores.append(r2_score(y_te, pred))
                        logger.info(f"Fold score for {name}: {scores[-1]:.3f}")
                    score = np.mean(scores) if scores else -1
                    if task == "classification":
                        results[name] = {"cv_f1_macro": score, "score": score}
                    else:
                        results[name] = {"cv_r2": score, "score": score}
                    logger.info(f"{name} - CV {task} Score: {score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                except Exception as e:
                    logger.error(f"Error with {name} in CV: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
                )
            except ValueError as e:
                logger.warning(f"Stratify failed: {e}. Falling back to no stratify with smaller test size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=42
                )
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"Train split - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Test split - X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    if name in ("SVM", "SVR", "KNN"):
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    model = builder(req.dict())
                    model.fit(X_tr, y_train)
                    pred = model.predict(X_te)
                    
                    if task == "classification":
                        acc = accuracy_score(y_test, pred)
                        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                        score = f1
                        results[name] = {"accuracy": acc, "f1_macro": f1, "score": score}
                        logger.info(f"{name} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
                    else:
                        r2 = r2_score(y_test, pred)
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        score = r2
                        results[name] = {"r2": r2, "mse": mse, "rmse": rmse, "score": score}
                        logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = model
                        best_pred = pred
                        y_test_best = y_test
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}

            if best_model:
                joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name.replace(' ', '_')}.joblib"))

        perf_plotly = create_performance_plot(results, task, best_model_name) if best_model_name and best_model_name != "No successful models" else None
        confusion_plotly = create_confusion_matrix(y_test_best, best_pred, best_model_name) if task == "classification" and best_pred is not None and len(np.unique(best_pred)) > 1 else None
        feature_importance_plotly = create_feature_importance_plot(best_model, X.columns.tolist(), best_model_name) if best_model and hasattr(X, "columns") and len(X.columns) > 0 else None
    else:
        for name, builder in MODEL_REGISTRY["unsupervised"].items():
            try:
                model = builder(req.dict())
                model.fit(X_scaled if name in ["KMeans", "PCA"] else X)
                if name == "KMeans":
                    results[name] = {"inertia": model.inertia_}
                elif name == "PCA":
                    results[name] = {"explained_variance": model.explained_variance_ratio_.tolist()}
            except Exception as e:
                logger.error(f"Error with {name} in unsupervised: {str(e)}")
                results[name] = {"error": str(e)}

    if best_model_name is None:
        successful_models = [name for name, result in results.items() if "error" not in result and "cv_" not in result]
        best_model_name = successful_models[0] if successful_models else "No successful models"
        best_score = results.get(best_model_name, {}).get("score", 0) if successful_models else 0

    df_stats = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "task": task,
        "target_unique": len(np.unique(y)) if task != "unsupervised" else 0,
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
    }
    
    # Simplified explanation without embeddings
    best_desc = MODEL_DESCRIPTIONS.get(best_model_name, "Analysis completed based on performance metrics.")
    perf = (
        f"achieving {results[best_model_name]['accuracy']:.1%} accuracy and {results[best_model_name]['f1_macro']:.3f} F1-macro"
        if task == "classification" and "accuracy" in results.get(best_model_name, {})
        else f"achieving R² = {results[best_model_name]['r2']:.3f}"
        if task == "regression" and "r2" in results.get(best_model_name, {})
        else f"achieving CV score of {best_score:.3f}"
    )
    explanation = f"Based on your dataset, {best_model_name} performed best, {perf}. {best_desc}"

    # Save results
    result_file = os.path.join(MODEL_DIR, f"train_result_{os.getpid()}.json")
    with open(result_file, "w") as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "perf_plotly": perf_plotly,
            "confusion_plotly": confusion_plotly,
            "feature_importance_plotly": feature_importance_plotly,
            "explanation": explanation,
            "dataset_stats": df_stats,
            "completed_at": pd.Timestamp.now().isoformat()
        }, f)
    logger.info(f"Training completed. Results saved to {result_file}")
    logger.info(f"Starting training for {req.upload_path} with target {req.target_column} at {pd.Timestamp.now()}")
    df = load_df(req.upload_path)
    
    # Validate target column exists
    if req.target_column not in df.columns:
        logger.error(f"Target column '{req.target_column}' not found in dataset")
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found in dataset")

    # Drop rows where target column has NaN
    df = df.dropna(subset=[req.target_column])
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data after dropping NaN values in target column")

    task = detect_task(df[req.target_column])
    
    X, y, X_scaled, target_encoder, scaler = preprocess(df, req.target_column)
    logger.info(f"Task: {task}, X shape: {X.shape}, y shape: {y.shape}")
    
    results = {}
    best_score = -np.inf
    best_model_name = None
    best_model = None
    best_pred = None
    y_test_best = None

    if task in ["classification", "regression"]:
        class_counts = np.bincount(y) if task == "classification" else np.ones(len(y), dtype=int)
        min_samples_per_class = np.min(class_counts[class_counts > 0]) if task == "classification" else len(y)
        total_samples = len(y)

        if total_samples < 20 or (task == "classification" and len(np.unique(y)) / total_samples > 0.5):
            logger.info("Dataset too small for train-test split. Using cross-validation or LOOCV.")
            if min_samples_per_class == 1 and task == "classification":
                logger.info("Using Leave-One-Out Cross-Validation.")
                cv = LeaveOneOut()
            else:
                n_splits = min(5, min_samples_per_class, total_samples)
                logger.info(f"Using {n_splits}-fold cross-validation.")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    model = builder(req.dict())
                    scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]
                        if name in ("SVM", "SVR", "KNN"):
                            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
                        model.fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        if task == "classification":
                            scores.append(f1_score(y_te, pred, average="macro", zero_division=0))
                        else:
                            scores.append(r2_score(y_te, pred))
                        logger.info(f"Fold score for {name}: {scores[-1]:.3f}")
                    score = np.mean(scores) if scores else -1
                    if task == "classification":
                        results[name] = {"cv_f1_macro": score, "score": score}
                    else:
                        results[name] = {"cv_r2": score, "score": score}
                    logger.info(f"{name} - CV {task} Score: {score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                except Exception as e:
                    logger.error(f"Error with {name} in CV: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
                )
            except ValueError as e:
                logger.warning(f"Stratify failed: {e}. Falling back to no stratify with smaller test size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=42
                )
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"Train split - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Test split - X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    if name in ("SVM", "SVR", "KNN"):
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    model = builder(req.dict())
                    model.fit(X_tr, y_train)
                    pred = model.predict(X_te)
                    
                    if task == "classification":
                        acc = accuracy_score(y_test, pred)
                        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                        score = f1
                        results[name] = {"accuracy": acc, "f1_macro": f1, "score": score}
                        logger.info(f"{name} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
                    else:
                        r2 = r2_score(y_test, pred)
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        score = r2
                        results[name] = {"r2": r2, "mse": mse, "rmse": rmse, "score": score}
                        logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = model
                        best_pred = pred
                        y_test_best = y_test
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}

            if best_model:
                joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name.replace(' ', '_')}.joblib"))

        perf_plotly = create_performance_plot(results, task, best_model_name) if best_model_name and best_model_name != "No successful models" else None
        confusion_plotly = create_confusion_matrix(y_test_best, best_pred, best_model_name) if task == "classification" and best_pred is not None and len(np.unique(best_pred)) > 1 else None
        feature_importance_plotly = create_feature_importance_plot(best_model, X.columns.tolist(), best_model_name) if best_model and hasattr(X, "columns") and len(X.columns) > 0 else None
    else:
        for name, builder in MODEL_REGISTRY["unsupervised"].items():
            try:
                model = builder(req.dict())
                model.fit(X_scaled if name in ["KMeans", "PCA"] else X)
                if name == "KMeans":
                    results[name] = {"inertia": model.inertia_}
                elif name == "PCA":
                    results[name] = {"explained_variance": model.explained_variance_ratio_.tolist()}
            except Exception as e:
                logger.error(f"Error with {name} in unsupervised: {str(e)}")
                results[name] = {"error": str(e)}

    if best_model_name is None:
        successful_models = [name for name, result in results.items() if "error" not in result and "cv_" not in result]
        best_model_name = successful_models[0] if successful_models else "No successful models"
        best_score = results.get(best_model_name, {}).get("score", 0) if successful_models else 0

    df_stats = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "task": task,
        "target_unique": len(np.unique(y)) if task != "unsupervised" else 0,
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
    }
    
    context = f"Dataset: {df_stats['rows']} rows, {df_stats['cols']} features. Task: {task}. "
    if task == "classification":
        context += f"Classes: {df_stats['target_unique']}. "
    context += f"Missing: {df_stats['missing_pct']:.1f}%."
    
    best_desc = MODEL_DESCRIPTIONS.get(best_model_name, "Analysis completed based on performance metrics.")

    perf = (
        f"achieving {results[best_model_name]['accuracy']:.1%} accuracy and {results[best_model_name]['f1_macro']:.3f} F1-macro"
        if task == "classification" and "accuracy" in results.get(best_model_name, {})
        else f"achieving R² = {results[best_model_name]['r2']:.3f}"
        if task == "regression" and "r2" in results.get(best_model_name, {})
        else f"achieving CV score of {best_score:.3f}"
    )
    explanation = f"Based on your dataset, {best_model_name} performed best, {perf}. {best_desc}"

    # Save results
    result_file = os.path.join(MODEL_DIR, f"train_result_{os.getpid()}.json")
    with open(result_file, "w") as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "perf_plotly": perf_plotly,
            "confusion_plotly": confusion_plotly,
            "feature_importance_plotly": feature_importance_plotly,
            "explanation": explanation,
            "dataset_stats": df_stats,
            "completed_at": pd.Timestamp.now().isoformat()
        }, f)
    logger.info(f"Training completed. Results saved to {result_file}")
    logger.info(f"Starting training for {req.upload_path} with target {req.target_column} at {pd.Timestamp.now()}")
    df = load_df(req.upload_path)
    
    # Validate target column exists
    if req.target_column not in df.columns:
        logger.error(f"Target column '{req.target_column}' not found in dataset")
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found in dataset")

    # Drop rows where target column has NaN
    df = df.dropna(subset=[req.target_column])
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data after dropping NaN values in target column")

    task = detect_task(df[req.target_column])
    
    X, y, X_scaled, target_encoder, scaler = preprocess(df, req.target_column)
    logger.info(f"Task: {task}, X shape: {X.shape}, y shape: {y.shape}")
    
    results = {}
    best_score = -np.inf
    best_model_name = None
    best_model = None
    best_pred = None
    y_test_best = None

    if task in ["classification", "regression"]:
        class_counts = np.bincount(y) if task == "classification" else np.ones(len(y), dtype=int)
        min_samples_per_class = np.min(class_counts[class_counts > 0]) if task == "classification" else len(y)
        total_samples = len(y)

        if total_samples < 20 or (task == "classification" and len(np.unique(y)) / total_samples > 0.5):
            logger.info("Dataset too small for train-test split. Using cross-validation or LOOCV.")
            if min_samples_per_class == 1 and task == "classification":
                logger.info("Using Leave-One-Out Cross-Validation.")
                cv = LeaveOneOut()
            else:
                n_splits = min(5, min_samples_per_class, total_samples)
                logger.info(f"Using {n_splits}-fold cross-validation.")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    model = builder(req.dict())
                    scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]
                        if name in ("SVM", "SVR", "KNN"):
                            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
                        model.fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        if task == "classification":
                            scores.append(f1_score(y_te, pred, average="macro", zero_division=0))
                        else:
                            scores.append(r2_score(y_te, pred))
                        logger.info(f"Fold score for {name}: {scores[-1]:.3f}")
                    score = np.mean(scores) if scores else -1
                    if task == "classification":
                        results[name] = {"cv_f1_macro": score, "score": score}
                    else:
                        results[name] = {"cv_r2": score, "score": score}
                    logger.info(f"{name} - CV {task} Score: {score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                except Exception as e:
                    logger.error(f"Error with {name} in CV: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
                )
            except ValueError as e:
                logger.warning(f"Stratify failed: {e}. Falling back to no stratify with smaller test size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.1, random_state=42
                )
            
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"Train split - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Test split - X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            for name, builder in MODEL_REGISTRY[task].items():
                try:
                    if name in ("SVM", "SVR", "KNN"):
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    model = builder(req.dict())
                    model.fit(X_tr, y_train)
                    pred = model.predict(X_te)
                    
                    if task == "classification":
                        acc = accuracy_score(y_test, pred)
                        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                        score = f1
                        results[name] = {"accuracy": acc, "f1_macro": f1, "score": score}
                        logger.info(f"{name} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
                    else:
                        r2 = r2_score(y_test, pred)
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        score = r2
                        results[name] = {"r2": r2, "mse": mse, "rmse": rmse, "score": score}
                        logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = model
                        best_pred = pred
                        y_test_best = y_test
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    results[name] = {"error": str(e), "score": -1}

            if best_model:
                joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name.replace(' ', '_')}.joblib"))

        perf_plotly = create_performance_plot(results, task, best_model_name) if best_model_name and best_model_name != "No successful models" else None
        confusion_plotly = create_confusion_matrix(y_test_best, best_pred, best_model_name) if task == "classification" and best_pred is not None and len(np.unique(best_pred)) > 1 else None
        feature_importance_plotly = create_feature_importance_plot(best_model, X.columns.tolist(), best_model_name) if best_model and hasattr(X, "columns") and len(X.columns) > 0 else None
    else:
        for name, builder in MODEL_REGISTRY["unsupervised"].items():
            try:
                model = builder(req.dict())
                model.fit(X_scaled if name in ["KMeans", "PCA"] else X)
                if name == "KMeans":
                    results[name] = {"inertia": model.inertia_}
                elif name == "PCA":
                    results[name] = {"explained_variance": model.explained_variance_ratio_.tolist()}
            except Exception as e:
                logger.error(f"Error with {name} in unsupervised: {str(e)}")
                results[name] = {"error": str(e)}

    if best_model_name is None:
        successful_models = [name for name, result in results.items() if "error" not in result and "cv_" not in result]
        best_model_name = successful_models[0] if successful_models else "No successful models"
        best_score = results.get(best_model_name, {}).get("score", 0) if successful_models else 0

    df_stats = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "task": task,
        "target_unique": len(np.unique(y)) if task != "unsupervised" else 0,
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
    }
    
    context = f"Dataset: {df_stats['rows']} rows, {df_stats['cols']} features. Task: {task}. "
    if task == "classification":
        context += f"Classes: {df_stats['target_unique']}. "
    context += f"Missing: {df_stats['missing_pct']:.1f}%."
    
    best_desc = MODEL_DESCRIPTIONS.get(best_model_name, "Analysis completed based on performance metrics.")

    perf = (
        f"achieving {results[best_model_name]['accuracy']:.1%} accuracy and {results[best_model_name]['f1_macro']:.3f} F1-macro"
        if task == "classification" and "accuracy" in results.get(best_model_name, {})
        else f"achieving R² = {results[best_model_name]['r2']:.3f}"
        if task == "regression" and "r2" in results.get(best_model_name, {})
        else f"achieving CV score of {best_score:.3f}"
    )
    explanation = f"Based on your dataset, {best_model_name} performed best, {perf}. {best_desc}"

    # Save results
    result_file = os.path.join(MODEL_DIR, f"train_result_{os.getpid()}.json")
    with open(result_file, "w") as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "perf_plotly": perf_plotly,
            "confusion_plotly": confusion_plotly,
            "feature_importance_plotly": feature_importance_plotly,
            "explanation": explanation,
            "dataset_stats": df_stats,
            "completed_at": pd.Timestamp.now().isoformat()
        }, f)
    logger.info(f"Training completed. Results saved to {result_file}")
class ReportRequest(BaseModel):
    upload_path: str
    target_column: str
    train_results: dict

@app.post("/generate_report_data")
def generate_report(req: ReportRequest):
    df = load_df(req.upload_path)
    report_metadata = {
        "dataset_name": os.path.basename(req.upload_path),
        "target_column": req.target_column,
        "rows": df.shape[0],
        "cols": df.shape[1]
    }
    categoricals = [col for col in df.columns if df[col].dtype == 'object' and col != req.target_column]
    gains = [id3_gain(df, col, req.target_column) for col in categoricals]
    id3_gain_analysis = {"columns": categoricals, "gains": gains}
    ml_analysis = {
        "best_model": req.train_results.get("best_model"),
        "results": req.train_results.get("results")
    }
    report = {
        "report_metadata": report_metadata,
        "id3_gain_analysis": id3_gain_analysis,
        "ml_analysis": ml_analysis
    }

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO reports (upload_path, target_column, train_results, report_data) VALUES (?, ?, ?, ?)",
              (req.upload_path, req.target_column, json.dumps(req.train_results), json.dumps(report)))
    report_id = c.lastrowid
    conn.commit()
    conn.close()

    return {"report_id": report_id}

@app.get("/report/{report_id}")
def get_report(report_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT report_data FROM reports WHERE id = ?", (report_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return JSONResponse(content=json.loads(row[0]))
    raise HTTPException(status_code=404, detail="Report not found")