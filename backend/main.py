# backend/main.py
import os, io, joblib, json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from ml_utils import load_df, preprocess

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load small embedding model (CPU friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Prewritten descriptions for basic semantic recommendations
MODEL_DESCRIPTIONS = {
    "DecisionTree": "Decision trees are simple, interpretable models. They handle categorical and numerical features, but they can overfit if deep.",
    "RandomForest": "Random forests are ensembles of trees. They reduce overfitting, work well with non-linear relations, and handle noise robustly."
}
# Precompute embeddings
MODEL_DESC_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k, v in MODEL_DESCRIPTIONS.items()}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), target_column: str = Form(None)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Replace NaN / Inf with None for JSON safety
        df = df.replace([np.nan, np.inf, -np.inf], None)

        target = target_column if target_column in df.columns else df.columns[-1]

        preview = df.head(5).to_dict(orient="records")
        stats = {
            "shape": df.shape,
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
            "n_missing": df.isnull().sum().to_dict(),
            "target": target,
        }

        # Save to temporary path
        save_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(contents)

        return JSONResponse(content={
            "ok": True,
            "preview": preview,
            "stats": stats,
            "upload_path": save_path
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )

class TrainRequest(BaseModel):
    upload_path: str
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int = None
    n_estimators: int = 100
    target_column: str = None

@app.post("/train")
def train(req: TrainRequest):
    df = pd.read_csv(req.upload_path)
    target_col = req.target_column if req.target_column in df.columns else df.columns[-1]
    X, y = preprocess(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state)

    # Train Decision Tree
    dt = DecisionTreeClassifier(max_depth=req.max_depth, random_state=req.random_state)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred, average='macro')

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=req.n_estimators, max_depth=req.max_depth, random_state=req.random_state)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')

    results = {
        "DecisionTree": {"accuracy": dt_acc, "f1_macro": dt_f1},
        "RandomForest": {"accuracy": rf_acc, "f1_macro": rf_f1}
    }
    # Pick best by f1_macro
    best_model_name = max(results.items(), key=lambda x: x[1]["f1_macro"])[0]
    best_model = dt if best_model_name == "DecisionTree" else rf
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name}.joblib"))

    # Create Plotly chart (bar)
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=list(results.keys()), y=[results[m]["accuracy"] for m in results]),
        go.Bar(name='F1 Macro', x=list(results.keys()), y=[results[m]["f1_macro"] for m in results])
    ])
    fig.update_layout(barmode='group', title="Model Performance")

    # Confusion matrix for best model
    best_pred = dt_pred if best_model_name == "DecisionTree" else rf_pred
    cm = confusion_matrix(y_test, best_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0]))))
    cm_fig.update_layout(title=f"Confusion Matrix ({best_model_name})")

    # Simple semantic "recommendation" using embeddings
    df_summary = f"rows:{df.shape[0]} cols:{df.shape[1]} numeric:{len(X.select_dtypes(include=['number']).columns)}"
    ds_emb = embedder.encode(df_summary, convert_to_tensor=True)
    sims = {k: float(util.cos_sim(ds_emb, MODEL_DESC_EMB[k])) for k in MODEL_DESCRIPTIONS.keys()}
    top = sorted(sims.items(), key=lambda x: x[1], reverse=True)[0]
    explanation = MODEL_DESCRIPTIONS[top[0]]
    explanation += f" (similarity {top[1]:.3f}). Recommendation: try max_depth={req.max_depth or 'None'} and n_estimators={req.n_estimators}."

    return {
        "ok": True,
        "results": results,
        "best_model": best_model_name,
        "perf_plotly": fig.to_json(),
        "cm_plotly": cm_fig.to_json(),
        "explanation": explanation
    }
