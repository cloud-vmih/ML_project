from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

# NUMERIC FEATURES
NUMERIC_FEATURES = [
    "year", "votes", "votes_log", "meta_score",
    "budget", "budget_log", "opening_weekend_gross", "opening_weekend_gross_log",
    "grossworldwwide", "grossworldwwide_log", "gross_us_canada", "gross_us_canada_log"
]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Hàm parse string list
def split_multi_label(df_col):
    series = df_col.iloc[:, 0]
    def parse(s):
        if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
            return []
        s = s.strip("[]").replace("'", "").replace('"', '')
        items = [item.strip() for item in s.split(",") if item.strip()]
        return items
    return series.apply(parse)

# Wrapper để bỏ qua y
class MultiLabelBinarizerIgnoreY(MultiLabelBinarizer):
    def fit_transform(self, X, y=None):
        return super().fit_transform(X)
    def transform(self, X, y=None):
        return super().transform(X)

def make_multilabel_pipeline():
    return Pipeline(steps=[
        ("split", FunctionTransformer(split_multi_label, validate=False)),
        ("mlb", MultiLabelBinarizerIgnoreY())
    ])

# Preprocessor chính
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("genres", make_multilabel_pipeline(), ["genres"]),
        ("languages", make_multilabel_pipeline(), ["languages"]),
        ("countries", make_multilabel_pipeline(), ["countries_origin"]),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
    sparse_threshold=0
)