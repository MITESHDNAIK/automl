# backend/ml_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_df(file_path):
    return pd.read_csv(file_path)

def infer_target_column(df, provided=None):
    if provided and provided in df.columns:
        return provided
    return df.columns[-1]

def preprocess(df, target_col):
    """Enhanced preprocessing for multiple algorithm types"""
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Handle missing values more robustly
    # Numerical columns - use median for better robustness
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    # Categorical columns - use most frequent
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # One-hot encode categoricals (with handling for high cardinality)
    for col in cat_cols:
        unique_values = X[col].nunique()
        # Limit high cardinality features to prevent curse of dimensionality
        if unique_values > 50:
            # Keep only top 10 most frequent categories, rest become 'Other'
            top_categories = X[col].value_counts().head(10).index
            X[col] = X[col].where(X[col].isin(top_categories), 'Other')
    
    # Apply one-hot encoding
    X = pd.get_dummies(X, drop_first=True, dummy_na=False)
    
    # Handle any remaining missing values that might have been introduced
    X = X.fillna(0)

    # Encode target if categorical
    target_encoder = None
    if y.dtype == object or y.dtype.name == "category":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
    # Scale features for algorithms that need it (SVM, KNN, Neural Networks)
    # We'll return both scaled and unscaled versions
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    return X, y, X_scaled, target_encoder, scaler

def detect_task(y):
    """Enhanced task detection with more sophisticated heuristics"""
    if pd.api.types.is_numeric_dtype(y):
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # If target has very few unique values relative to dataset size, it's likely classification
        if unique_values <= 10 or unique_values / total_values < 0.05:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

def get_algorithm_recommendations(X, y, task):
    """Recommend best algorithms based on dataset characteristics"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y)) if task == "classification" else None
    
    recommendations = []
    
    # Dataset size considerations
    if n_samples < 1000:
        recommendations.extend(["Naive Bayes", "KNN", "Decision Tree"])
    elif n_samples < 10000:
        recommendations.extend(["Random Forest", "SVM", "XGBoost"])
    else:
        recommendations.extend(["XGBoost", "Random Forest", "Linear Regression" if task == "regression" else "Logistic Regression"])
    
    # Feature count considerations
    if n_features > n_samples:  # High-dimensional data
        if task == "classification":
            recommendations.extend(["Logistic Regression", "Naive Bayes", "SVM"])
        else:
            recommendations.extend(["Linear Regression", "SVR"])
    
    # Classification specific
    if task == "classification":
        if n_classes == 2:  # Binary classification
            recommendations.extend(["Logistic Regression", "SVM", "XGBoost"])
        else:  # Multi-class
            recommendations.extend(["Random Forest", "XGBoost", "Naive Bayes"])
    
    return list(set(recommendations))  # Remove duplicates

def evaluate_data_quality(df, target_col):
    """Evaluate dataset quality and provide insights"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    quality_report = {
        "missing_data_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "target_balance": y.value_counts().to_dict() if y.dtype == 'object' or len(y.unique()) <= 10 else None,
        "feature_types": {
            "numerical": len(X.select_dtypes(include=[np.number]).columns),
            "categorical": len(X.select_dtypes(exclude=[np.number]).columns)
        },
        "high_cardinality_features": [col for col in X.select_dtypes(exclude=[np.number]).columns if X[col].nunique() > 20],
        "duplicated_rows": df.duplicated().sum(),
        "constant_features": [col for col in X.columns if X[col].nunique() <= 1]
    }
    
    return quality_report