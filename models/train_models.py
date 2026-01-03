import numpy as np
import pandas as pd
import pickle
import sys
import os

# Thêm src vào path
sys.path.append('src')

# Import custom models
from models.base_model.linear_regression import RidgeRegressionCustom
from models.base_model.random_forest import RandomForestCustom
from models.base_model.knn import KNNRegressorCustom
from models.stacking import ManualStackingEnsemble as StackingEnsembleCustom

def train_and_save_models():
    """Train và save các custom models"""
    
    print("🎯 TRAINING CUSTOM MODELS FROM SCRATCH")
    
    # Tạo dữ liệu giả (thay bằng data thật của bạn)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] * 2 + 
               X_train[:, 1] * 1.5 + 
               np.sin(X_train[:, 2]) + 
               np.random.randn(n_samples) * 0.5)
    
    # Tạo models
    ridge_model = RidgeRegressionCustom(alpha=1.0, fit_intercept=True)
    rf_model = RandomForestCustom(n_estimators=50, max_depth=8)
    knn_model = KNNRegressorCustom(k=7, weights='distance')
    
    # Train base models
    print("\n1. Training Ridge Regression...")
    ridge_model.fit(X_train, y_train)
    
    print("2. Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    print("3. Training KNN...")
    knn_model.fit(X_train, y_train)
    
    # Tạo và train Stacking Ensemble
    print("\n4. Training Stacking Ensemble...")
    base_models = [ridge_model, rf_model, knn_model]
    meta_model = RidgeRegressionCustom(alpha=0.5, fit_intercept=True)
    
    stacking_model = StackingEnsembleCustom(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=5
    )
    
    stacking_model.fit(X_train, y_train)
    
    # Save models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n💾 Saving models to '{models_dir}/'...")
    
    with open(f'{models_dir}/ridge_custom.pkl', 'wb') as f:
        pickle.dump(ridge_model, f)
    
    with open(f'{models_dir}/rf_custom.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(f'{models_dir}/knn_custom.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    
    with open(f'{models_dir}/stacking_custom.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    
    print("✅ Models saved successfully!")
    
    # Test predictions
    X_test = np.random.randn(5, n_features)
    
    print("\n🧪 Test predictions:")
    for i, x in enumerate(X_test):
        ridge_pred = ridge_model.predict([x])[0]
        rf_pred = rf_model.predict([x])[0]
        knn_pred = knn_model.predict([x])[0]
        stacking_pred = stacking_model.predict([x])[0]
        
        print(f"Sample {i}: Ridge={ridge_pred:.3f}, RF={rf_pred:.3f}, "
              f"KNN={knn_pred:.3f}, Stacking={stacking_pred:.3f}")

if __name__ == "__main__":
    train_and_save_models()