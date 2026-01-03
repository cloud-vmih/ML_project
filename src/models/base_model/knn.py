import numpy as np
from collections import Counter
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class KNNRegressorCustom:
    def __init__(self, k='auto', weights='uniform', metric='euclidean', k_range=(1, 20)):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.k_range = k_range
        self.X_train = None
        self.y_train = None
        self.best_k = None
        self.cv_results_ = None
        
    def _distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _find_optimal_k_cv(self, X, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        k_values = range(self.k_range[0], self.k_range[1] + 1)
        
        # Lưu kết quả cho từng k
        k_scores = []
        
        for k in k_values:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Train KNN với k hiện tại trên fold
                knn_fold = KNNRegressorCustom(k=k, weights=self.weights, metric=self.metric)
                knn_fold.fit(X_train_fold, y_train_fold)
                
                # Dự đoán và tính score
                y_pred = knn_fold.predict(X_val_fold)
                
                # Tính RMSE
                rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
                fold_scores.append(rmse)
            
            # Trung bình score trên tất cả folds
            avg_score = np.mean(fold_scores)
            k_scores.append(avg_score)
        
        # Chọn k với score tốt nhất (RMSE nhỏ nhất)
        best_idx = np.argmin(k_scores)
        best_k = k_values[best_idx]
        
        # Lưu kết quả
        self.cv_results_ = {
            'k_values': list(k_values),
            'scores': k_scores,
            'best_k': best_k,
            'best_score': k_scores[best_idx]
        }
        return best_k
    
    def _find_optimal_k_sqrt(self, n_samples):
        """Quy tắc heuristic: sqrt(n_samples)"""
        k = int(np.sqrt(n_samples))
        return max(3, min(k, 20))  # Giới hạn trong khoảng 3-20
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Lưu training data
        self.X_train = X
        self.y_train = y
        
        # Tự động tìm K tối ưu nếu cần
        if self.k == 'auto':
            n_samples = len(X)
            
            #self.best_k = self._find_optimal_k_cv(X, y, cv=min(5, n_samples // 10))
            self.best_k = self._find_optimal_k_sqrt(n_samples)
            
            print(f"Auto-selected k: {self.best_k}")
        else:
            self.best_k = self.k
        
        return self
    
    def _predict_single(self, x_test):
        """Dự đoán cho một điểm dữ liệu"""
        # Tính khoảng cách đến tất cả training points (vectorized)
        distances = self._distance(x_test, self.X_train)
        
        # Tìm k nearest neighbors
        self.best_k = int(self.best_k)
        k = min(self.best_k, len(distances))
        indices = np.argpartition(distances, k - 1)[:self.best_k]
        k_distances = distances[indices]
        k_labels = self.y_train[indices]
        
        # Tính prediction
        if self.weights == 'uniform':
            pred = np.mean(k_labels)
        elif self.weights == 'distance':
            weights = 1 / (k_distances + 1e-8)
            pred = np.average(k_labels, weights=weights)
        else:
            raise ValueError(f"Unknown weight: {self.weights}")
        
        return pred
    
    def predict(self, X):
        """Dự đoán cho nhiều điểm dữ liệu"""
        X = np.array(X)
        
        if self.X_train is None:
            raise ValueError("Model chưa được huấn luyện. Gọi fit() trước.")
        
        # Vectorized prediction
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def get_params(self, deep=True):
        return {
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'k_range': self.k_range,
            'best_k': self.best_k
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y).flatten()
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    def plot_k_selection(self):
        if self.cv_results_ is None:
            print("Không có dữ liệu chọn K. Chạy fit() với k='auto' trước.")
            return None
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Error vs K
        axes[0].plot(self.cv_results_['k_values'], self.cv_results_.get('errors', self.cv_results_.get('scores', [])), 
                    'b-o', linewidth=2, markersize=8)
        axes[0].axvline(self.cv_results_['best_k'], color='r', linestyle='--', 
                       label=f'Best k = {self.cv_results_["best_k"]}')
        axes[0].set_xlabel('Number of Neighbors (k)', fontsize=12)
        axes[0].set_ylabel('Error' if 'errors' in self.cv_results_ else 'CV Score', fontsize=12)
        axes[0].set_title('Error vs Number of Neighbors', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        plt.tight_layout()
        return fig
    
    def _batch_predict(self, X, batch_size=1000):
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X[i:end_idx]
            predictions[i:end_idx] = [self._predict_single(x) for x in batch]
        
        return predictions

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
from data.preprocess_for_model.knn import knn_preprocessor, KNN_FEATURES, engineer_knn_features
if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv("data/split/train.csv")
    test_df  = pd.read_csv("data/split/test.csv")

    # Lấy nhãn rating
    y_train = train_df["rating"].values.reshape(-1, 1)
    y_test  = test_df["rating"].values.reshape(-1, 1)
    

    X_train_df = train_df.drop(columns=["rating"])
    X_test_df  = test_df.drop(columns=["rating"])

    X_train_df = engineer_knn_features(X_train_df)
    X_test_df  = engineer_knn_features(X_test_df)

    X_train = knn_preprocessor.fit_transform(X_train_df[KNN_FEATURES])
    X_test  = knn_preprocessor.transform(X_test_df[KNN_FEATURES])
    
    print(f"Đã tải train: {X_train.shape[0]} mẫu")
    print(f"Đã tải test : {X_test.shape[0]} mẫu\n")
        
    print("\n2. KNN với k='auto' (tự động chọn):")
    knn_auto = KNNRegressorCustom(k='auto', weights='distance')
    knn_auto.fit(X_train, y_train)
    y_pred_auto = knn_auto.predict(X_test)
    mse_auto = np.mean((y_test - y_pred_auto) ** 2)
    print(f"   Best k selected: {knn_auto.best_k}")
    print(f"   MSE: {mse_auto:.4f}")
    print(f"   R² Score: {knn_auto.score(X_test, y_test):.4f}")
    
    # Test 3: So sánh các metrics
    print("\n3. So sánh các distance metrics:")
    for metric in ['euclidean', 'manhattan']:
        knn = KNNRegressorCustom(k=7, weights='distance', metric=metric)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        print(f"   {metric}: R² = {score:.4f}")
    
    # Hiển thị biểu đồ chọn K nếu có
    if knn_auto.cv_results_ is not None:
        print("\n5. Kết quả chọn K:")
        print(f"K tối ưu: {knn_auto.cv_results_['best_k']}")
        print(f"Error/Score tại K tối ưu: {knn_auto.cv_results_.get('best_score', knn_auto.cv_results_.get('best_error', 'N/A')):.4f}")
        
        # Vẽ biểu đồ
        fig = knn_auto.plot_k_selection()
        if fig:
            import matplotlib.pyplot as plt
            plt.show()
 
    
    print("\nTesting completed")