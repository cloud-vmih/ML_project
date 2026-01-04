import numpy as np
import pandas as pd
import sys
import os

from sklearn.model_selection import KFold


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from data.preprocess_for_model import linear_preprocessor

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=2000, random_state=42, auto = True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.auto = auto
        self.random_state = random_state
        self.w = None  # Vector trọng số
        self.b = None  # b
    def choose_hyperparameters(self, cv = 5, X=None, y=None, lr_range=(0.001, 0.1)):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        lr_values = np.arange(lr_range[0], lr_range[1], 0.001)
        
        # Lưu kết quả cho từng k
        lr_scores = []
        
        for lr in lr_values:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train TRỰC TIẾP không gọi fit() để tránh đệ quy
                linear_fold = LinearRegressionGD(learning_rate=lr, epochs=500, random_state=self.random_state, auto=False)
                linear_fold.fit(X_train_fold, y_train_fold)
                y_pred_val = linear_fold.predict(X_val_fold)
                rmse = np.sqrt(np.mean((y_val_fold.reshape(-1, 1) - y_pred_val) ** 2))
                fold_scores.append(rmse)
            
            avg_score = np.mean(fold_scores)
            lr_scores.append(avg_score)
            print(f"   LR: {lr:.4f} | Avg RMSE: {avg_score:.4f}")
        
        # Chọn lr tốt nhất
        best_idx = np.argmin(lr_scores)
        best_lr = lr_values[best_idx]
        
        self.cv_results_ = {
            'lr_values': lr_values.tolist(),
            'scores': lr_scores,
            'best_lr': best_lr,
            'best_score': lr_scores[best_idx]
        }
        
        print(f"Learning rate tối ưu: {best_lr:.4f}")
        self.learning_rate = best_lr
        return best_lr
    def fit(self, X, y):
        np.random.seed(self.random_state)
        if self.auto == True:
            lr = self.choose_hyperparameters(X=X, y=y)
        else:
            lr = self.learning_rate
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features, 1) * lr
        self.b = 0.0


        # print("Learning Rate được chọn:", lr)
        # print(f"Bắt đầu huấn luyện với {n_samples} mẫu và {n_features} đặc trưng...")
        # print("-" * 70)

        for epoch in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.w -= lr * dw
            self.b -= lr * db

            if epoch % 100 == 0 and self.auto == True:
                loss = np.mean(error ** 2) / 2
                print(f"Epoch {epoch:4d} | Loss = {loss:.6f}")

        # in Loss
        if self.auto == True:
            final_loss = np.mean((X @ self.w + self.b - y) ** 2) / 2
            print(f"Epoch {self.epochs:4d} | Loss = {final_loss:.6f}")
            print("-" * 70)
            print("Huấn luyện hoàn tất!\n")

    def predict(self, X):
        return X @ self.w + self.b

    @staticmethod
    def evaluate(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return mae, rmse, r2
    def get_params(self):
        return {
            'LR': self.learning_rate,
            'epochs': self.epochs,
            'weights': self.w,
            'bias': self.b
        }

#Train thử
def main():
    #load data
    train_df = pd.read_csv("data/split/train.csv")
    test_df  = pd.read_csv("data/split/test.csv")

    print(f"Đã tải train: {train_df.shape[0]} mẫu")
    print(f"Đã tải test : {test_df.shape[0]} mẫu\n")

    # Lấy nhãn rating
    y_train = train_df["rating"].values.reshape(-1, 1)
    y_test  = test_df["rating"].values.reshape(-1, 1)

    X_train_df = train_df.drop(columns=["rating"])
    X_test_df  = test_df.drop(columns=["rating"])

    X_train = linear_preprocessor.fit_transform(X_train_df)
    X_test  = linear_preprocessor.transform(X_test_df)
    
    print(f"Kích thước sau preprocess - Train: {X_train.shape} | Test: {X_test.shape}\n")

    # huấn luyện
    model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=2000,
        random_state=42,
        auto=True
    )

    model.fit(X_train, y_train)

    print("=" * 70)
    print(f"{'HỆ SỐ MÔ HÌNH HUẤN LUYỆN':^70}")
    print("=" * 70)
    print(f" Hệ số b: {model.b.item():.8f}")
    print("-" * 70)
    print("Vector trọng số (w):")
    print(model.w.flatten())
    print("=" * 70)

    #dự đoán và đánh giá
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_mae, train_rmse, train_r2 = model.evaluate(y_train, y_train_pred)
    test_mae,  test_rmse,  test_r2  = model.evaluate(y_test, y_test_pred)

    #in kết quả đánh giá
    print(f"{'ĐÁNH GIÁ MÔ HÌNH LINEAR REGRESSION (Gradient Descent)':^70}")
    print("=" * 70)
    print(f"{'Metric':<10} | {'Train':<20} | {'Test':<20}")
    print("-" * 70)
    print(f"{'MAE':<10} | {train_mae:<20.4f} | {test_mae:<20.4f}")
    print(f"{'RMSE':<10} | {train_rmse:<20.4f} | {test_rmse:<20.4f}")
    print(f"{'R² Score':<10} | {train_r2:<20.4f} | {test_r2:<20.4f}")
    print("=" * 70)

    #Phân tích overfitting/underfitting
    print("\n" + "=" * 70)
    print(f"{'PHÂN TÍCH HIỆU SUẤT MÔ HÌNH':^70}")
    print("=" * 70)

    delta_rmse = test_rmse - train_rmse
    delta_r2   = train_r2 - test_r2

    print(f"Chênh lệch RMSE (Test - Train): {delta_rmse:+.4f}")
    print(f"Chênh lệch R²   (Train - Test): {delta_r2:+.4f}")

    if delta_rmse > 0.3 or delta_r2 > 0.1:
        print("\nKết luận: Mô hình có dấu hiệu OVERFITTING nhẹ")
    elif train_rmse > 1.0 or train_r2 < 0.4:
        print("\nKết luận: Mô hình có dấu hiệu UNDERFITTING nhẹ")
    else:
        print("\nKết luận: Mô hình không overfit/underfit rõ rệt")

    print("   → Linear Regression phù hợp làm mô hình cơ sở (baseline).")
    print("   → Có thể cải thiện bằng mô hình phức tạp hơn (Random Forest, ...).")
    print("=" * 70)


if __name__ == "__main__":
    main()