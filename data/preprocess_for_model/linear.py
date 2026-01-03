# data/preprocess_for_model/linear.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Danh sách các cột numerical
NUMERIC_FEATURES = [
    "duration",
    "votes_log",
    "meta_score",
    "budget_log",
    "grossworldwwide_log",
    "gross_us_canada_log",             
    "opening_weekend_gross_log",        
    # Count features 
    "num_genres",
    "num_languages",
    "num_stars",
    "num_awards"
]

# Danh sách categorical
CATEGORICAL_FEATURES = ["mpa"]

# ==================== feature engineering - count features ====================
def add_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các count features từ metadata để tăng thông tin cho mô hình dự đoán rating
    """
    df = df.copy()

    # Số lượng genres
    if "genres" in df.columns:
        df["num_genres"] = df["genres"].apply(
            lambda x: len(eval(x)) if pd.notna(x) and isinstance(x, str) and x.startswith("[")
            else len(str(x).split(",")) if pd.notna(x) else 0
        )

    # Số lượng languages
    if "languages" in df.columns:
        df["num_languages"] = df["languages"].apply(
            lambda x: len(eval(x)) if pd.notna(x) and isinstance(x, str) and x.startswith("[")
            else len(str(x).split(",")) if pd.notna(x) else 0
        )

    # Số lượng stars (diễn viên chính)
    if "stars" in df.columns:
        df["num_stars"] = df["stars"].apply(
            lambda x: len(eval(x)) if pd.notna(x) and isinstance(x, str) and x.startswith("[")
            else len(str(x).split(",")) if pd.notna(x) else 0
        )

    # Số lượng awards (đếm từ khóa win/nomination)
    if "awards_content" in df.columns:
        df["num_awards"] = df["awards_content"].apply(
            lambda x: str(x).lower().count("win") + str(x).lower().count("nomination")
            if pd.notna(x) and str(x) != "No information" else 0
        )

    return df


# pipeline xử lý 
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Final preprocessor
linear_preprocessor = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(add_count_features)),
    ("preprocessing", ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"  
    ))
])