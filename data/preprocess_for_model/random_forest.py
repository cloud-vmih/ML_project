from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer
 

# Mục tiêu:
# không scale
# giữ numeric 
# multihot cho genre 

NUMERIC_FEATURES = [
    "duration",
    "votes_log",
    "meta_score",
    "budget_log",
    "grossworldwwide_log"
]

GENRE_FEATURE = ["genres"]
def split_genres(x):
    return x.fillna("").apply(lambda v: v.split(","))

genre_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("split", FunctionTransformer(split_genres)),
    ("mlb", MultiLabelBinarizer())
])

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

#final preprocessor for random forest models
rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("genre", genre_pipeline, GENRE_FEATURE),
    ],
    remainder="drop"
)
