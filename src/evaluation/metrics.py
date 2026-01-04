import numpy as np

class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()

    def mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

    def rmse(self):
        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))

    def r2(self):
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)

    def to_dict(self):
        return {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "R²": self.r2()
        }
