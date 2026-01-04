import numpy as np
from models.meta_model.ridge_regression import RidgeRegressionMetaModel
import pandas as pd

class ManualStackingEnsemble:  
    def __init__(self, base_models, meta_model=None, n_folds=5):
        self.base_models = base_models
        self.n_base_models = len(base_models)
        self.n_folds = n_folds
        
        # Sử dụng Ridge Regression làm meta-model mặc định
        if meta_model is None:
            print("0")
            self.meta_model = RidgeRegressionMetaModel(alpha=0.5, fit_intercept=True)
        else:
            print("1")
            self.meta_model = meta_model
            
        self.trained_base_models = []
        self.meta_features_train = None
        
    def _k_fold_split(self, X, y, k):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_size = n_samples // k
        folds = []
        
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n_samples
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            folds.append((train_indices, val_indices))
            
        return folds
    def _safe_index(self, X, idx):
        if hasattr(X, "iloc"):      # pandas
            return X.iloc[idx]
        else:                       # numpy
            return X[idx]
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        #BƯỚC 1: Tạo meta-features bằng k-fold cross-validation
        print("\nBƯỚC 1: Tạo meta-features bằng {self.n_folds}-fold CV")
        
        #Khởi tạo matrix cho meta-features
        self.meta_features_train = np.zeros((n_samples, self.n_base_models))
        
        #Tạo folds
        folds = self._k_fold_split(X, y, self.n_folds)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
            print(f"\n  Fold {fold_idx}/{self.n_folds}:")
            
            X_train_fold = self._safe_index(X, train_idx)
            X_val_fold   = self._safe_index(X, val_idx)

            y_train_fold = self._safe_index(y, train_idx)
            y_val_fold   = self._safe_index(y, val_idx)
            
            for model_idx, model in enumerate(self.base_models):
                #Train base model trên fold training data
                model.fit(X_train_fold, y_train_fold)
                
                #Predict trên fold validation data
                y_pred_val = model.predict(X_val_fold)
                
                #Lưu predictions vào meta-features matrix
                self.meta_features_train[val_idx, model_idx] = y_pred_val.flatten()
                
                print(f"Model {model_idx+1}: {len(y_pred_val)} predictions")
            print(f"Model predictions for this fold {self.meta_features_train[val_idx]}")
        
        #BƯỚC 2: Huấn luyện base models trên toàn bộ data
        print("\nBƯỚC 2: Huấn luyện base models trên toàn bộ dataset")
        
        self.trained_base_models = []
        for model_idx, model in enumerate(self.base_models):
            model.fit(X, y)  # Train trên toàn bộ data
            self.trained_base_models.append(model)
            print(f"  Model {model_idx+1}: Trained on {len(X)} samples")
        
        #BƯỚC 3: Huấn luyện meta-model
        print("\nBƯỚC 3: Huấn luyện meta-model (Ridge Regression)")
        
        self.meta_model.fit(self.meta_features_train, y, method='closed_form')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        
        return self
    
    def predict(self, X):
        #Bước 1: Dự đoán từ base models
        base_predictions = []
        
        for model in self.trained_base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        #Stack predictions theo chiều ngang
        meta_features_test = np.column_stack(base_predictions)
        
        #Bước 2: Dự đoán từ meta-model
        final_predictions = self.meta_model.predict(meta_features_test)
        
        return final_predictions
    
    def get_stacking_summary(self, ensemble_r2):
        importance = self.meta_model.get_feature_importance()
        base_names = [f"Model_{i+1}" for i in range(self.n_base_models)]
        formula = self.meta_model.get_final_formula(base_names)

        summary = {
            "meta_model": {
                "type": "Ridge Regression",
                "alpha": self.meta_model.alpha,
                "fit_intercept": self.meta_model.fit_intercept,
                "meta_r2": self.meta_model.r2_score  
            },
            "ensemble": {
                "r2_score": ensemble_r2,
                "formula": formula
            },
            "weights": importance,
            "models": base_names
        }

        return summary
    
            
