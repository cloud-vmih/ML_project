import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


def engineer_knn_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # budget efficiency
    df["budget_to_gross"] = df["grossworldwwide"] / (df["budget"] + 1)
    df["budget_to_gross"] = np.log1p(df["budget_to_gross"])

    # votes theo thời gian
    CURRENT_YEAR = 2025
    df["votes_per_year"] = df["votes"] / (CURRENT_YEAR - df["year"] + 1)
    df["votes_per_year"] = np.log1p(df["votes_per_year"])

    # runtime bucket
    def runtime_bucket(x):
        if pd.isna(x):
            return np.nan
        if x < 90:
            return 0
        elif x < 120:
            return 1
        elif x < 150:
            return 2
        else:
            return 3

    df["runtime_category"] = df["duration"].apply(runtime_bucket)

    # meta score flag
    df["high_meta"] = (df["meta_score"] >= 70).astype(int)

    return df

KNN_FEATURES = [
    "votes_log",
    "votes_per_year",
    "duration",
    "runtime_category",
    "meta_score",
    "high_meta",
    "budget_log",
    "grossworldwwide_log",
    "budget_to_gross",
    "year"
]

def select_knn_features(df):
    return df[KNN_FEATURES]

knn_preprocessor = Pipeline(steps=[
    (
        "feature_engineering",
        FunctionTransformer(engineer_knn_features, validate=False)
    ),
    (
        "select_features",
        FunctionTransformer(select_knn_features, validate=False)
    ),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
