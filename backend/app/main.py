import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Pydantic model for training request
class TrainRequest(BaseModel):
    upload_path: str
    target_column: str
    max_depth: int | None = None
    n_estimators: int = 100
    kernel: str = "rbf"
    n_neighbors: int = 5
    n_clusters: int = 3
    n_components: int = 2

# Function to detect task type based on target column
def detect_task(df, target_column):
    target_dtype = df[target_column].dtype
    if target_dtype in [np.float64, np.int64]:
        return "regression"
    return "classification"

# Function to prepare data with categorical encoding
def prepare_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.fillna(0)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X, y

# Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), target_column: str = Form(None)):
    try:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        df = pd.read_csv(file_path)
        if target_column and target_column in df.columns:
            pass  # Target already set
        elif not target_column and df.columns[-1] not in ["index", "Unnamed: 0"]:
            target_column = df.columns[-1]
        else:
            target_column = df.columns[0]  # Default to first column if no obvious target
        logger.info(f"No target column specified. Defaulting to {target_column}")
        
        stats = {
            "shape": list(df.shape),
            "target": target_column,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},  # Convert dtypes to strings
            "n_missing": df.isnull().sum().to_dict()  # Already JSON-serializable
        }
        # Handle NaN values in numerical_data_for_plot by replacing with null
        numerical_data_for_plot = {
            col: [None if pd.isna(x) else x for x in df[col].tolist()]
            for col in df.select_dtypes(include=[np.number]).columns if col != target_column
        }
        
        upload_info = {
            "upload_path": file_path,
            "stats": stats,
            "numerical_data_for_plot": numerical_data_for_plot
        }
        return upload_info
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Entropy gain endpoint (placeholder)
@app.post("/entropy_gain")
async def entropy_gain():
    return {"message": "Entropy gain calculation not implemented"}

# Train endpoint
@app.post("/train")
async def train(req: TrainRequest):
    task_id = str(uuid.uuid4())
    df = pd.read_csv(req.upload_path)
    X, y = prepare_data(df, req.target_column)
    
    # Run training in background
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(train_in_background(task_id, df, req))
    return {"task_id": task_id, "status": "pending"}

async def train_in_background(task_id: str, df: pd.DataFrame, req: TrainRequest):
    try:
        logger.info(f"Starting training for {req.upload_path} with target {req.target_column} at {datetime.now()}")
        task = detect_task(df, req.target_column)  # Use target column for task detection
        X, y = prepare_data(df, req.target_column)
        logger.info(f"Task: {task}, X shape: {X.shape}, y shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train split - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Test split - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=req.max_depth),
            "Random Forest": RandomForestRegressor(n_estimators=req.n_estimators),
            "SVR": SVR(kernel=req.kernel),
            "KNN": KNeighborsRegressor(n_neighbors=req.n_neighbors)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task == "regression":
                from sklearn.metrics import r2_score, mean_squared_error
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {"R²": round(r2, 3), "RMSE": round(rmse, 3)}
                logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            else:
                from sklearn.metrics import accuracy_score, f1_score
                accuracy = accuracy_score(y_test, y_pred.round())
                f1 = f1_score(y_test, y_pred.round(), average="macro")
                results[name] = {"Accuracy": round(accuracy, 3), "F1 Macro": round(f1, 3)}
                logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1 Macro: {f1:.3f}")
        
        best_model = max(results.items(), key=lambda x: list(x[1].values())[0])[0]
        result = {
            "status": "completed",
            "best_model": best_model,
            "results": results
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        result = {
            "status": "failed",
            "error": str(e),
            "results": {}
        }
    
    result_path = os.path.join(MODELS_DIR, f"train_result_{task_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    logger.info(f"Training completed. Results saved to {result_path}")

# Train result endpoint
@app.get("/train_result/{task_id}")
async def train_result(task_id: str):
    result_path = os.path.join(MODELS_DIR, f"train_result_{task_id}.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)
    return {"status": "pending"}

# Generate report data endpoint (placeholder)
@app.post("/generate_report_data")
async def generate_report_data():
    return {"message": "Report generation not implemented"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)