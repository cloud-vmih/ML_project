import numpy as np
import pandas as pd
import random
from math import sqrt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)
from data.preprocess_for_model import rf_preprocessor

# TREE NODE
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf value

    def is_leaf(self):
        return self.value is not None

# DECISION TREE REGRESSOR
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        y = np.array(y)
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features_
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return TreeNode(value=np.mean(y))

        feature_indices = np.random.choice(
            n_features, self.max_features, replace=False
        )
        best_feature, best_threshold, best_mse = None, None, float("inf")
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                mse = self._weighted_mse(y[left_mask], y[right_mask])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold
        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _weighted_mse(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (
            len(y_left) / n * self._mse(y_left)
            + len(y_right) / n * self._mse(y_right)
        )

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

# RANDOM FOREST REGRESSOR
class RandomForestRegressor:
    def __init__(
        self,
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        np.random.seed(random_state)
        random.seed(random_state)

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        y = np.array(y)
        self.trees = []
        n_samples, n_features = X.shape
        if self.max_features == "sqrt":
            max_features = int(sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap(X, y)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)

# ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def percentage_within_error(y_true, y_pred, threshold):
    return np.mean(np.abs(y_true - y_pred) <= threshold) * 100

def main():
    print("Loading data...")
    train_df = pd.read_csv(r"C:\MLProj\ML_project\data\split\train.csv")
    test_df  = pd.read_csv(r"C:\MLProj\ML_project\data\split\test.csv")

    X_train = rf_preprocessor.fit_transform(train_df)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    y_train = train_df["rating"].values

    X_test = rf_preprocessor.transform(test_df)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    y_test = test_df["rating"].values

    print(f"Số mẫu train: {len(y_train)} | Số mẫu test: {len(y_test)}")

    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print("Predicting on test set...")
    y_pred = rf.predict(X_test)

    print("          ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH TRÊN TẬP TEST")
    print(f"RMSE                  : {rmse(y_test, y_pred):.4f}")
    print(f"MAE                   : {mae(y_test, y_pred):.4f}")
    print(f"R² Score              : {r2_score(y_test, y_pred):.4f}")
    print(f"Tỷ lệ |error| ≤ 0.5   : {percentage_within_error(y_test, y_pred, 0.5):.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()