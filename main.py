from src.preproccess import concat_data, data_cleaning
import os
import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split
# Thêm src vào path
sys.path.append('src')

# Import custom models
from models.base_model.linear_regression import RidgeRegressionCustom
from models.base_model.random_forest import RandomForestCustom
from models.base_model.knn import KNNRegressorCustom
from models.stacking import ManualStackingEnsemble as StackingEnsembleCustom

def main():
    # Gom dữ liệu phim từ các file CSV
    base_path = f"{os.getcwd()}/data/raw/Data/"
    output_path = f"{os.getcwd()}/data/raw/all_movies_data.csv"
    concat_data.concat_movie_data(base_path=base_path, output_path=output_path)
    
    df = pd.read_csv(output_path) 
    # (Khuyến nghị) ép kiểu Year để tránh lỗi
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # Lọc dữ liệu theo năm (1975–2025)
    df = df[(df["Year"] >= 1975) & (df["Year"] <= 2025)]
    # Lưu lại dữ liệu sau khi lọc
    filtered_output_path = f"{os.getcwd()}/data/raw/all_movies_data_1975_2025.csv"
    df.to_csv(filtered_output_path, index=False)

    # Tiền xử lý dữ liệu
    clean_output_path = f"{os.getcwd()}/data/processed/clean_movies_data_1975_2025.csv"
    data_cleaning.clean_data(file_path=filtered_output_path, output_path=clean_output_path)
    
    # Huấn luyện và lưu các custom models
    """Train và save các custom models"""
    
    print("🎯 TRAINING CUSTOM MODELS FROM SCRATCH")
    
    df = pd.read_csv(clean_output_path)
    
    X_train = df.drop(columns=["rating"]).values
    y_train = df["rating"].values
    
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Tạo models
    ridge_model = RidgeRegressionCustom(alpha=1.0, fit_intercept=True)
    rf_model = RandomForestCustom(n_estimators=50, max_depth=8)
    knn_model = KNNRegressorCustom(k=7, weights='distance')
    
    # Train base models
    print("\n1. Training Ridge Regression...")
    ridge_model.fit(X_train, y_train)
    
    print("2. Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Train và lưu các custom models
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
    
    print(f"\nSaving models to '{models_dir}/'...")
    
    with open(f'{models_dir}/ridge_custom.pkl', 'wb') as f:
        pickle.dump(ridge_model, f)
    
    with open(f'{models_dir}/rf_custom.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(f'{models_dir}/knn_custom.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    
    with open(f'{models_dir}/stacking_custom.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    
    print("Models saved successfully!")
    
    print("\nTest predictions:")
    for i, x in enumerate(X_val):
        ridge_pred = ridge_model.predict([x])[0]
        rf_pred = rf_model.predict([x])[0]
        knn_pred = knn_model.predict([x])[0]
        stacking_pred = stacking_model.predict([x])[0]
        
        print(f"Sample {i}: Ridge={ridge_pred:.3f}, RF={rf_pred:.3f}, "
              f"KNN={knn_pred:.3f}, Stacking={stacking_pred:.3f}")

if __name__ == "__main__":
    main()