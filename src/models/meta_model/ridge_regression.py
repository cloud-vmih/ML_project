import numpy as np

class RidgeRegressionMetaModel:
    """
    Ridge Regression Meta-Model implement từ đầu
    Dùng để combine predictions từ base models
    """
    
    def __init__(self, alpha=0.5, fit_intercept=True):
        """
        Khởi tạo Ridge Regression Meta-Model
        
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
            alpha càng lớn, regularization càng mạnh
        fit_intercept : bool
            Có thêm bias term (intercept) hay không
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.weights = None  # Vector trọng số w
        self.bias = 0.0  # Bias term (nếu có)
        
    def _add_intercept(self, X):
        """Thêm cột 1 cho intercept (bias term)"""
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X
    
    def _remove_intercept(self):
        """Tách bias từ weights nếu có intercept"""
        if self.fit_intercept and self.weights is not None:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
    
    def _closed_form_solution(self, X, y):
        """
        Giải Ridge Regression bằng công thức đóng
        
        Công thức: w = (X^T X + αI)^(-1) X^T y
        
        Returns:
            weights: Vector trọng số
        """
        n_samples, n_features = X.shape
        
        # X^T X
        XTX = X.T @ X
        
        # Tạo ma trận identity I
        I = np.eye(n_features)
        
        # Nếu có intercept, không regularize bias term
        if self.fit_intercept:
            I[0, 0] = 0  # Không regularize cột intercept
        
        # (X^T X + αI)
        XTX_regularized = XTX + self.alpha * I
        
        # (X^T X + αI)^(-1) X^T y
        try:
            # Inverse của ma trận
            weights = np.linalg.inv(XTX_regularized) @ X.T @ y
        except np.linalg.LinAlgError:
            # Nếu ma trận singular, dùng pseudoinverse
            print("Warning: Matrix is singular, using pseudoinverse")
            weights = np.linalg.pinv(XTX_regularized) @ X.T @ y
        
        return weights
    
    def _gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Giải Ridge Regression bằng Gradient Descent
        
        Loss function: L(w) = ||y - Xw||² + α||w||²
        Gradient: ∇L(w) = -2X^T(y - Xw) + 2αw
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Lưu loss để theo dõi
        losses = []
        
        for i in range(n_iterations):
            # Dự đoán
            y_pred = X @ self.weights
            
            # Tính loss (MSE + regularization)
            mse = np.mean((y - y_pred) ** 2)
            reg_term = self.alpha * np.sum(self.weights ** 2)
            loss = mse + reg_term
            losses.append(loss)
            
            # Tính gradient
            # Gradient của MSE: -2 * X^T(y - Xw) / n_samples
            mse_grad = -2 * X.T @ (y - y_pred) / n_samples
            
            # Gradient của regularization: 2αw
            reg_grad = 2 * self.alpha * self.weights
            
            # Gradient tổng
            gradient = mse_grad + reg_grad
            
            # Update weights
            self.weights -= learning_rate * gradient
            
            # In loss mỗi 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")
        
        return losses
    
    def fit(self, X_meta, y, method='closed_form', **kwargs):
        """
        Huấn luyện Ridge Regression Meta-Model
        
        Parameters:
        -----------
        X_meta : numpy array, shape (n_samples, n_base_models)
            Meta-features từ base models predictions
        y : numpy array, shape (n_samples,)
            Target values (ratings)
        method : str
            Phương pháp giải: 'closed_form' hoặc 'gradient_descent'
        **kwargs : dict
            Tham số cho gradient descent (learning_rate, n_iterations)
        """
        # Chuyển đổi thành numpy array
        X_meta = np.array(X_meta, dtype=np.float64)
        y = np.array(y, dtype=np.float64).flatten()
        
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
            weights = self._closed_form_solution(X_with_intercept, y)
            self.weights = weights
            
            # Tách bias nếu có intercept
            if self.fit_intercept:
                self.bias = weights[0]
                self.weights = weights[1:]
                
        elif method == 'gradient_descent':
            # Giải bằng gradient descent
            learning_rate = kwargs.get('learning_rate', 0.01)
            n_iterations = kwargs.get('n_iterations', 1000)
            
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
            print(f"  Bias: {self.bias:.4f}")
        
        return self
    
    def predict(self, X_meta):
        """
        Dự đoán với meta-model
        
        Parameters:
        -----------
        X_meta : numpy array, shape (n_samples, n_base_models)
            Predictions từ base models
        
        Returns:
        --------
        y_pred : numpy array
            Dự đoán cuối cùng
        """
        X_meta = np.array(X_meta, dtype=np.float64)
        
        # Dự đoán: y = Xw + b
        y_pred = X_meta @ self.weights
        
        if self.fit_intercept:
            y_pred += self.bias
            
        return y_pred
    
    def _calculate_r2(self, y_true, y_pred):
        """Tính R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Thêm epsilon để tránh chia 0
        return r2
    
    def get_feature_importance(self):
        """
        Lấy độ quan trọng của từng base model
        
        Returns:
        --------
        importance : dict
            Dictionary với base model và weight tương ứng
        """
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
        """
        Lấy công thức cuối cùng của meta-model
        
        Parameters:
        -----------
        base_model_names : list
            Tên của các base models
        
        Returns:
        --------
        formula : str
            Công thức dự đoán
        """
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


# ============================================================================
# STACKING PIPELINE HOÀN CHỈNH VỚI RIDGE META-MODEL
# ============================================================================





# # ============================================================================
# # DEMO VÀ TEST
# # ============================================================================

# def test_ridge_regression():
#     """Test Ridge Regression Meta-Model"""
#     print("\n" + "="*60)
#     print("TEST RIDGE REGRESSION META-MODEL")
#     print("="*60)
    
#     # Tạo dữ liệu giả lập: 3 base models, 100 samples
#     np.random.seed(42)
#     n_samples = 100
#     n_models = 3
    
#     # Tạo predictions từ base models
#     X_meta = np.random.randn(n_samples, n_models) * 0.5
    
#     # Tạo target: kết hợp tuyến tính của predictions + noise
#     true_weights = np.array([0.3, 0.5, 0.2])
#     bias = 0.5
#     y = X_meta @ true_weights + bias + np.random.randn(n_samples) * 0.1
    
#     # Tạo và train Ridge Regression meta-model
#     ridge_model = RidgeRegressionMetaModel(alpha=0.1, fit_intercept=True)
#     ridge_model.fit(X_meta, y, method='closed_form')
    
#     # Dự đoán
#     y_pred = ridge_model.predict(X_meta)
    
#     # Đánh giá
#     mse = np.mean((y - y_pred) ** 2)
#     r2 = ridge_model.r2_score
    
#     print(f"\nResults:")
#     print(f"  True weights: {true_weights}")
#     print(f"  True bias: {bias}")
#     print(f"  Learned weights: {ridge_model.weights}")
#     print(f"  Learned bias: {ridge_model.bias}")
#     print(f"  MSE: {mse:.6f}")
#     print(f"  R²: {r2:.4f}")
    
#     # Feature importance
#     importance = ridge_model.get_feature_importance()
#     print(f"\nFeature Importance:")
#     for model, weight in importance.items():
#         print(f"  {model}: {weight:.4f}")
    
#     # Final formula
#     formula = ridge_model.get_final_formula(['f₁(x)', 'f₂(x)', 'f₃(x)'])
#     print(f"\nFinal Formula:")
#     print(f"  {formula}")
    
#     return ridge_model

# def test_stacking_ensemble():
#     """Test toàn bộ Stacking Ensemble"""
#     print("\n" + "="*60)
#     print("TEST STACKING ENSEMBLE")
#     print("="*60)
    
#     # Tạo dữ liệu giả lập
#     np.random.seed(42)
#     n_samples = 200
#     n_features = 5
    
#     # Features
#     X = np.random.randn(n_samples, n_features)
    
#     # Target: hàm phi tuyến
#     y = (X[:, 0] ** 2 + np.sin(X[:, 1]) + 
#          X[:, 2] * X[:, 3] + np.random.randn(n_samples) * 0.5)
    
#     # Tạo base models
#     base_models = [
#         SimpleLinearModel(),
#         SimpleKNN(k=7),
#         SimpleDecisionTree(max_depth=4)
#     ]
    
#     # Tạo Stacking Ensemble
#     stacking = ManualStackingEnsemble(
#         base_models=base_models,
#         meta_model=RidgeRegressionMetaModel(alpha=0.5, fit_intercept=True),
#         n_folds=3
#     )
    
#     # Train
#     stacking.fit(X, y)
    
#     # Predict
#     y_pred = stacking.predict(X)
    
#     # Đánh giá
#     mse = np.mean((y - y_pred) ** 2)
#     print(f"\nStacking Ensemble Performance:")
#     print(f"  MSE: {mse:.6f}")
    
#     # So sánh với base models
#     print(f"\nBase Models Performance:")
#     for i, model in enumerate(base_models):
#         model_pred = model.predict(X)
#         model_mse = np.mean((y - model_pred) ** 2)
#         print(f"  Model {i+1}: MSE = {model_mse:.6f}")
    
#     # Summary
#     stacking.get_model_summary()
    
#     return stacking

# if __name__ == "__main__":
#     print("🎯 IMPLEMENTING RIDGE REGRESSION META-MODEL FROM SCRATCH")
    
#     # Test 1: Ridge Regression Meta-Model
#     ridge_model = test_ridge_regression()
    
#     # Test 2: Full Stacking Ensemble
#     stacking_model = test_stacking_ensemble()
    
#     print("\n" + "="*60)
#     print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
#     print("="*60)