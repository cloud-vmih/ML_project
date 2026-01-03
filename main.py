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
from models.base_model.linear_regression import LinearRegressionGD 
from models.base_model.random_forest import RandomForestCustom
from models.base_model.knn import KNNRegressorCustom
from models.stacking import ManualStackingEnsemble as StackingEnsembleCustom
from models.meta_model.ridge_regression import RidgeRegressionMetaModel
from data.preprocess_for_model import linear_preprocessor
from data.preprocess_for_model.knn import knn_preprocessor, KNN_FEATURES, engineer_knn_features

    
def main():
    # # Gom dữ liệu phim từ các file CSV
    # base_path = f"{os.getcwd()}/data/raw/Data/"
    # output_path = f"{os.getcwd()}/data/raw/all_movies_data.csv"
    # concat_data.concat_movie_data(base_path=base_path, output_path=output_path)
    
    # df = pd.read_csv(output_path) 
    # # (Khuyến nghị) ép kiểu Year để tránh lỗi
    # df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # # Lọc dữ liệu theo năm (1975–2025)
    # df = df[(df["Year"] >= 1975) & (df["Year"] <= 2025)]
    # # Lưu lại dữ liệu sau khi lọc
    # filtered_output_path = f"{os.getcwd()}/data/raw/all_movies_data_1975_2025.csv"
    # df.to_csv(filtered_output_path, index=False)

    # # Tiền xử lý dữ liệu
    # clean_output_path = f"{os.getcwd()}/data/processed/clean_movies_data_1975_2025.csv"
    # data_cleaning.clean_data(file_path=filtered_output_path, output_path=clean_output_path)
    
    # Huấn luyện và lưu các custom models
    train_df = pd.read_csv("data/split/train.csv")
    test_df  = pd.read_csv("data/split/test.csv")
    
    y_train = train_df["rating"].values.reshape(-1, 1)
    y_test  = test_df["rating"].values.reshape(-1, 1)

    X_train_df = train_df.drop(columns=["rating"])
    X_test_df  = test_df.drop(columns=["rating"])
    
    # Tạo models
    linear_model = LinearRegressionGD(learning_rate=0.01, epochs=2000)
    rf_model = RandomForestCustom(n_estimators=50, max_depth=8)
    knn_model = KNNRegressorCustom(weights='distance', k_range=(5,30))
    
    # Train base models
    print("\n1. Training Linear Regression...")
    X_train = linear_preprocessor.fit_transform(X_train_df)
    X_test  = linear_preprocessor.transform(X_test_df)
    linear_model.fit(X_train, y_train)
    
    # print("2. Training Random Forest...")
    # rf_model.fit(X_train_df, y_train)
    
    # Train và lưu các custom models
    print("3. Training KNN...")
    X_train = engineer_knn_features(X_train)
    X_test  = engineer_knn_features(X_test)
    X_train = knn_preprocessor.fit_transform(X_train[KNN_FEATURES])
    X_test  = knn_preprocessor.transform(X_test[KNN_FEATURES])
    knn_model.fit(X_train, y_train)
    
    # Tạo và train Stacking Ensemble
    print("\n4. Training Stacking Ensemble...")
    base_models = [linear_model, knn_model]
    meta_model = RidgeRegressionMetaModel(alpha=0.5, fit_intercept=True)
    
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

    with open(f'{models_dir}/linear_custom.pkl', 'wb') as f:
        pickle.dump(linear_model, f)

    # with open(f'{models_dir}/rf_custom.pkl', 'wb') as f:
    #     pickle.dump(rf_model, f)
    
    with open(f'{models_dir}/knn_custom.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    
    with open(f'{models_dir}/stacking_custom.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    
    print("Models saved successfully!")
    
    print("\nTest predictions:")
    for i, x in enumerate(X_test[:5]):
        linear_pred = linear_model.predict([x])[0]
        #rf_pred = rf_model.predict([x])[0]
        knn_pred = knn_model.predict([x])[0]
        stacking_pred = stacking_model.predict([x])[0]

        print(f"Sample {i}: Linear={linear_pred:.3f}, "
              f"KNN={knn_pred:.3f}, Stacking={stacking_pred:.3f}")

if __name__ == "__main__":
    main()