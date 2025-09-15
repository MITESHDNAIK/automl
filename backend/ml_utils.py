# backend/ml_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_df(file_path):
    return pd.read_csv(file_path)

def infer_target_column(df, provided=None):
    if provided and provided in df.columns:
        return provided
    return df.columns[-1]

def preprocess(df, target_col):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Fill missing numeric with mean
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    # Fill missing categorical with mode
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "")

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if categorical
    if y.dtype == object or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        # numeric target (could be regression) — for MVP we focus on classification
        y = y

    return X, y
