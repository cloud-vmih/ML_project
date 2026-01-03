import numpy as np

class RegressionMetrics:
    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error
        MAE = (1/n) * Σ |y_i - ŷ_i|
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        """
        Root Mean Squared Error
        RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)^2 )
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2(y_true, y_pred):
        """
        R^2 = 1 - SS_res / SS_tot
        SS_res = Σ (y_i - ŷ_i)^2
        SS_tot = Σ (y_i - ȳ)^2
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # tránh chia cho 0
        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot
