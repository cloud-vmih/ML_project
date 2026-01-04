import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class RidgeRegressionMetaModel:
    def __init__(self, alpha=0.5, fit_intercept=True, auto = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.weights = None  # Vector trọng số w
        self.bias = 0.0  # Bias 
        self.scaler = StandardScaler()
        self.auto = auto
        self.best_alpha = None

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X
    
    def _remove_intercept(self):
        if self.fit_intercept and self.weights is not None:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
            
    def choose_hyperparameters(self, cv = 5, X=None, y=None):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        alpha_values = np.linspace(0.1, 10.0, 20)
        alpha_scores = []
        
        for alpha in alpha_values:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                ridge_fold = RidgeRegressionMetaModel(alpha=alpha, fit_intercept=self.fit_intercept, auto=False)
                ridge_fold.fit(X_train_fold, y_train_fold, method='closed_form')
                y_pred_val = ridge_fold.predict(X_val_fold)
                rmse = np.sqrt(np.mean((y_val_fold.reshape(-1, 1) - y_pred_val) ** 2))
                fold_scores.append(rmse)
            
            avg_score = np.mean(fold_scores)
            alpha_scores.append(avg_score)
            print(f"Alpha: {alpha:.4f} | Avg RMSE: {avg_score:.4f}")
        
        # Chọn alpha tốt nhất
        best_idx = np.argmin(alpha_scores)
        best_alpha = alpha_values[best_idx]
        
        self.cv_results_ = {
            'alpha_values': alpha_values.tolist(),
            'scores': alpha_scores,
            'best_alpha': best_alpha,
            'best_score': alpha_scores[best_idx]
        }
        
        print(f"Alpha tối ưu: {best_alpha:.4f}")
        self.best_alpha = best_alpha
        return best_alpha
        
    def _closed_form_solution(self, X, y):
        n_samples, n_features = X.shape
        # X^T X
        XTX = X.T @ X
        
        # Tạo ma trận identity I
        I = np.eye(n_features)
        
        # Nếu có intercept, không regularize bias term
        if self.fit_intercept:
            I[0, 0] = 0  
        
        # (X^T X + αI)
        XTX_regularized = XTX + self.alpha * I
        
        # (X^T X + αI)^(-1) X^T y
        try:
            # Nghiệm của ma trận
            weights = np.linalg.solve(XTX_regularized, X.T @ y)
        except np.linalg.LinAlgError:
            # Nếu ma trận singular, dùng pseudoinverse
            print("Warning: Matrix is singular, using pseudoinverse")
            weights = np.linalg.pinv(XTX_regularized) @ X.T @ y
        
        return weights
    
    def _gradient_descent(self, X, y, learning_rate=0.001, n_iterations=3000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Lưu loss để theo dõi
        losses = []
        
        for i in range(n_iterations):
            # Dự đoán
            y_pred = X @ self.weights
            
            # Tính loss (MSE + regularization)
            mse = np.mean((y - y_pred) ** 2)
            reg_term = self.alpha * np.sum(self.weights[1:] ** 2)
            loss = mse + reg_term
            losses.append(loss)
            
            # Tính gradient
            # Gradient của MSE: 2 * X^T(y - Xw) / n_samples
            mse_grad = 2 * X.T @ (y_pred - y) / n_samples
            
            # Gradient của regularization: 2αw
            reg_grad = np.zeros_like(self.weights) 
            reg_grad[1:] = 2 * self.alpha * self.weights[1:]
            
            # Gradient tổng
            gradient = mse_grad + reg_grad
            
            # Update weights
            self.weights -= learning_rate * gradient
            
            # In loss mỗi 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")
        
        return losses
    
    def fit(self, X_meta, y, method='closed_form', **kwargs):
        # Chuyển đổi thành numpy array
        X_meta = np.array(X_meta, dtype=np.float64)
        y = np.array(y, dtype=np.float64).flatten()
        X_meta = self.scaler.fit_transform(X_meta)
        # Thêm intercept nếu cần
        X_with_intercept = self._add_intercept(X_meta)
        
        # Lưu số base models
        self.n_base_models = X_meta.shape[1]
        
        print(f"Training Ridge Meta-Model (α={self.alpha})...")
        print(f"  Samples: {X_meta.shape[0]}")
        print(f"  Base models: {self.n_base_models}")
        print(f"  Method: {method}")
        
        if method == 'closed_form':
            # Giải bằng công thức đóng
            if self.auto == True:
                self.alpha = self.choose_hyperparameters(X=X_meta, y=y)
            else:
                self.alpha = self.alpha
            weights = self._closed_form_solution(X_with_intercept, y)
            self.weights = weights
            
            # Tách bias nếu có intercept
            if self.fit_intercept:
                self.bias = weights[0]
                self.weights = weights[1:]
                
        elif method == 'gradient_descent':
            # Giải bằng gradient descent
            learning_rate = kwargs.get('learning_rate', 0.001)
            n_iterations = kwargs.get('n_iterations', 3000)
            
            losses = self._gradient_descent(
                X_with_intercept, y, 
                learning_rate, n_iterations
            )
            
            # Tách bias nếu có intercept
            if self.fit_intercept:
                self.bias = self.weights[0]
                self.weights = self.weights[1:]
            
            self.loss_history = losses
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'closed_form' or 'gradient_descent'")
        
        # Tính R² score
        y_pred = self.predict(X_meta)
        self.r2_score = self._calculate_r2(y, y_pred)
        
        print(f"  Training completed!")
        print(f"  R² score: {self.r2_score:.4f}")
        print(f"  Weights: {self.weights}")
        if self.fit_intercept:
            print(f"Bias: {self.bias:.4f}")
        
        return self
    
    def predict(self, X_meta):
        X_meta = np.array(X_meta, dtype=np.float64)
        X_meta = self.scaler.transform(X_meta)
        # Dự đoán: y = Xw + b
        y_pred = X_meta @ self.weights
        
        if self.fit_intercept:
            y_pred += self.bias
            
        return y_pred
    
    def _calculate_r2(self, y_true, y_pred):
        """Tính R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  
        return r2
    
    def get_feature_importance(self):
        if self.weights is None:
            raise ValueError("Model chưa được trained")
        
        importance = {}
        for i, weight in enumerate(self.weights):
            model_name = f"Base_Model_{i+1}"
            importance[model_name] = float(weight)
        
        # Normalize để tổng absolute weights = 1
        total_abs = np.sum(np.abs(list(importance.values())))
        if total_abs > 0:
            for model in importance:
                importance[model] = importance[model] / total_abs
        
        return importance
    
    def get_final_formula(self, base_model_names=None):
        if base_model_names is None:
            base_model_names = [f"f{i+1}(x)" for i in range(self.n_base_models)]
        
        formula_parts = []
        
        # Thêm bias term nếu có
        if self.fit_intercept and abs(self.bias) > 1e-10:
            formula_parts.append(f"{self.bias:.4f}")
        
        # Thêm các base model terms
        for i, (name, weight) in enumerate(zip(base_model_names, self.weights)):
            if abs(weight) > 1e-10:  # Chỉ thêm nếu weight đáng kể
                sign = "+" if weight >= 0 else "-"
                abs_weight = abs(weight)
                formula_parts.append(f"{sign} {abs_weight:.4f}×{name}")
        
        formula = "ŷ = " + " ".join(formula_parts)
        return formula
    