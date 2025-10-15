import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# tiny embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_REGISTRY = {
    "classification": {
        "Decision Tree": lambda p: DecisionTreeClassifier(
            max_depth=p.get("max_depth") if p.get("max_depth") else None, random_state=42
        ),
        "Random Forest": lambda p: RandomForestClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42,
        ),
        "Logistic Regression": lambda p: LogisticRegression(max_iter=2000, random_state=42),
        "SVM": lambda p: SVC(kernel=p.get("kernel", "rbf"), random_state=42),
        "KNN": lambda p: KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5)),
        "Naive Bayes": lambda p: GaussianNB(),
    },
    "regression": {
        "Linear Regression": lambda p: LinearRegression(),
        "Decision Tree": lambda p: DecisionTreeRegressor(
            max_depth=p.get("max_depth") if p.get("max_depth") else None, random_state=42
        ),
        "Random Forest": lambda p: RandomForestRegressor(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth") if p.get("max_depth") else None,
            random_state=42,
        ),
        "SVR": lambda p: SVR(kernel=p.get("kernel", "rbf")),
        "KNN": lambda p: KNeighborsRegressor(n_neighbors=p.get("n_neighbors", 5)),
    },
    "unsupervised": {
        "KMeans": lambda p: KMeans(n_clusters=p.get("n_clusters", 3), random_state=42, n_init=10),
        "PCA": lambda p: PCA(n_components=p.get("n_components", 2)),
    },
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": "Best for linear relationships between features and target.",
    "Logistic Regression": "Excellent for binary classification tasks.",
    "Decision Tree": "Highly interpretable with hierarchical decision rules.",
    "Random Forest": "Robust ensemble method reducing overfitting.",
    "SVM": "Powerful for high-dimensional data with complex boundaries.",
    "KNN": "Simple instance-based learning algorithm.",
    "Naive Bayes": "Fast probabilistic classifier based on Bayes' theorem.",
    "KMeans": "Unsupervised clustering into k groups.",
    "PCA": "Dimensionality reduction preserving variance.",
    "SVR": "Support-vector regression for non-linear predictions.",
}
MODEL_DESC_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k, v in MODEL_DESCRIPTIONS.items()}