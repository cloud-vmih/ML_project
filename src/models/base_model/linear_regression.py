import numpy as np

class RidgeRegressionCustom:
    """Ridge Regression implement từ đầu"""
    
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Thêm intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        n_samples, n_features = X.shape
        
        # Ridge closed-form solution
        XTX = X.T @ X
        I = np.eye(n_features)
        if self.fit_intercept:
            I[0, 0] = 0  # Không regularize bias
        
        XTX_regularized = XTX + self.alpha * I
        
        try:
            weights = np.linalg.inv(XTX_regularized) @ X.T @ y
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(XTX_regularized) @ X.T @ y
        
        # Tách bias
        if self.fit_intercept:
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.weights = weights
            self.bias = 0.0
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def get_params(self):
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'weights': self.weights,
            'bias': self.bias
        }