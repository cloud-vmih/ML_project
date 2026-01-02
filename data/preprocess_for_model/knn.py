from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# cần scale 

KNN_FEATURES = [
    "votes_log",
    "duration",
    "meta_score",
    "budget_log",
    "grossworldwwide_log"
]

knn_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])