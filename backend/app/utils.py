import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

def init_db(db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  upload_path TEXT,
                  target_column TEXT,
                  train_results TEXT,
                  report_data TEXT)''')
    conn.commit()
    conn.close()

def load_df(file_path):
    return pd.read_csv(file_path)

def preprocess(df, target_column):
    df = df.copy()
    target = df[target_column]
    df = df.drop(columns=[target_column])
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])  # For categorical
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numerical_cols])
    X = pd.DataFrame(X_scaled, columns=numerical_cols)
    
    target_encoder = None
    if detect_task(target) == "classification":
        target_encoder = pd.Categorical(target).codes
        y = target_encoder
    else:
        y = target.values
    
    return X, y, X_scaled, target_encoder, scaler

def detect_task(target):
    unique_values = target.nunique()
    total_values = len(target)
    if unique_values == 2 or (unique_values / total_values < 0.05 and unique_values > 2):
        return "classification"
    elif np.issubdtype(target.dtype, np.number):
        return "regression"
    return "unsupervised"

def id3_gain(df, feature, target):
    total_entropy = entropy(df[target].value_counts(normalize=True))
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_entropy = entropy(df[df[feature] == value][target].value_counts(normalize=True))
        weighted_entropy += (count / len(df)) * subset_entropy
    return total_entropy - weighted_entropy

def create_feature_importance_plot(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        import plotly.graph_objects as go
        importances = model.feature_importances_
        fig = go.Figure(data=[go.Bar(x=feature_names, y=importances)])
        fig.update_layout(title=f'Feature Importance ({model_name})', xaxis_title='Features', yaxis_title='Importance')
        return fig.to_json()
    return None

# Placeholder for other plot functions (implement as needed)
def create_performance_plot(results, task, best_model_name):
    return None

def create_confusion_matrix(y_test, y_pred, model_name):
    return None