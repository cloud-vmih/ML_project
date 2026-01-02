from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

# NUMERIC FEATURES (NO SCALING)
NUMERIC_FEATURES = [
    "year",
    "rating",
    "votes",
    "votes_log",
    "meta_score",
    "budget",
    "budget_log",
    "opening_weekend_gross",
    "opening_weekend_gross_log",
    "grossworldwwide",
    "grossworldwwide_log",
    "gross_us_canada",
    "gross_us_canada_log"
]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# MULTI-LABEL FEATURES
def split_multi_label(series):
    return series.fillna("").apply(lambda x: [i.strip() for i in x.split(",") if i.strip()])

def make_multilabel_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("split", FunctionTransformer(split_multi_label)),
        ("mlb", MultiLabelBinarizer())
    ])

MULTI_LABEL_FEATURES = ["genres", "languages", "countries_origin"]

rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("genres", make_multilabel_pipeline(), "genres"),
        ("languages", make_multilabel_pipeline(), "languages"),
        ("countries", make_multilabel_pipeline(), "countries_origin"),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)
