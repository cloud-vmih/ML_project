import numpy as np
import pandas as pd

from data.preprocess_for_model.linear import linear_preprocessor

# =============================
# 1. LOAD DATA
# =============================
train_df = pd.read_csv("../data/raw/train.csv")
test_df  = pd.read_csv("../data/raw/test.csv")

# Target
y_train = train_df["Rating"].values.reshape(-1, 1)
y_test  = test_df["Rating"].values.reshape(-1, 1)

# Features
X_train_df = train_df.drop(columns=["Rating"])
X_test_df  = test_df.drop(columns=["Rating"])

# =============================
# 2. PREPROCESS (PIPELINE)
# =============================
X_train = linear_preprocessor.fit_transform(X_train_df)
X_test  = linear_preprocessor.transform(X_test_df)

n_samples, n_features = X_train.shape
print(f"Train samples: {n_samples}, Features: {n_features}")

# =============================
# 3. INIT LINEAR REGRESSION
# =============================
np.random.seed(42)
w = np.random.randn(n_features, 1) * 0.01
b = 0.0

learning_rate = 0.01
epochs = 2000

print("🚀 Bắt đầu huấn luyện Linear Regression...")

# =============================
# 4. TRAIN (GRADIENT DESCENT)
# =============================
for epoch in range(epochs):

    # Forward
    y_pred = X_train @ w + b

    # Loss
    error = y_pred - y_train
    loss = np.mean(error ** 2) / 2

    # Gradients
    dw = (1 / n_samples) * (X_train.T @ error)
    db = (1 / n_samples) * np.sum(error)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss:.6f}")

print("\n✅ HUẤN LUYỆN HOÀN TẤT")

# =============================
# 5. EVALUATION
# =============================
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return mae, rmse, r2

y_train_pred = X_train @ w + b
y_test_pred  = X_test @ w + b

train_mae, train_rmse, train_r2 = calculate_metrics(y_train, y_train_pred)
test_mae, test_rmse, test_r2    = calculate_metrics(y_test, y_test_pred)

print("\n" + "=" * 65)
print(f"{'Metric':<15} | {'Train':<20} | {'Test':<20}")
print("-" * 65)
print(f"{'MAE':<15} | {train_mae:<20.4f} | {test_mae:<20.4f}")
print(f"{'RMSE':<15} | {train_rmse:<20.4f} | {test_rmse:<20.4f}")
print(f"{'R2':<15} | {train_r2:<20.4f} | {test_r2:<20.4f}")
print("=" * 65)
