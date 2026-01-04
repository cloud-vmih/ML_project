from evaluation.metrics import RegressionMetrics
from src.preproccess import concat_data, data_cleaning
import os
import pandas as pd
import numpy as np
import sys
import pickle, json
from sklearn.pipeline import Pipeline
# Thêm src vào path
sys.path.append('src')

# Import custom models
from models.base_model.linear_regression import LinearRegressionGD 
from models.base_model.random_forest import RandomForestRegressor
from models.base_model.knn import KNNRegressorCustom
from models.stacking import ManualStackingEnsemble as StackingEnsembleCustom
from models.meta_model.ridge_regression import RidgeRegressionMetaModel
from data.preprocess_for_model import linear_preprocessor
from data.preprocess_for_model import knn_preprocessor
from data.preprocess_for_model import rf_preprocessor

    
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
    # # Save models
    models_dir = 'models_trained'
    os.makedirs(models_dir, exist_ok=True)
    # Tạo models
    linear_model = LinearRegressionGD(learning_rate=0.01, epochs=2000, auto=True)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, max_features="sqrt")
    knn_model = KNNRegressorCustom(weights='distance', k='auto', k_range=(5, 30))
    
    #Train base models
    print("\n1. Training Linear Regression...")
    X_train_linear = linear_preprocessor.fit_transform(X_train_df)
    X_test_linear  = linear_preprocessor.transform(X_test_df)
    linear_model.fit(X_train_linear, y_train)
    y_pred_linear = linear_model.predict(X_test_linear)
    print(f"\nSaving models to '{models_dir}/'...")

    with open(f'{models_dir}/linear_custom.pkl', 'wb') as f:
        pickle.dump(linear_model, f)
        
    metrics_linear = RegressionMetrics(y_test, y_pred_linear).to_dict()
    with open("models/linear_metrics.json", "w") as f:
        json.dump(metrics_linear, f, indent=4)
    
    print("2. Training KNN...")
    X_train_knn = knn_preprocessor.fit_transform(X_train_df)
    X_test_knn  = knn_preprocessor.transform(X_test_df)
    knn_model.fit(X_train_knn, y_train)
    y_pred_knn = knn_model.predict(X_test_knn)
    print(f"\nSaving models to '{models_dir}/'...")
    with open(f'{models_dir}/knn_custom.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    metrics_knn = RegressionMetrics(y_test, y_pred_knn).to_dict()
    with open("models/knn_metrics.json", "w") as f:
        json.dump(metrics_knn, f, indent=4)
    
    print("3. Training Random Forest...")
    X_train_rf = rf_preprocessor.fit_transform(X_train_df)
    X_test_rf = rf_preprocessor.transform(X_test_df)
    rf_model.fit(X_train_rf, y_train)
    y_pred_rf = rf_model.predict(X_test_rf)
    print(f"\nSaving models to '{models_dir}/'...")
    with open(f'{models_dir}/rf_custom.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    metrics_rf = RegressionMetrics(y_test, y_pred_rf).to_dict()
    with open("models/rf_metrics.json", "w") as f:
        json.dump(metrics_rf, f, indent=4)

    #Tạo và train Stacking Ensemble
    print("\n4. Training Stacking Ensemble...")
    #Đã tìm hyperparameters tốt cho base model Linear Regression trong stacking
    linear_model_stacking = LinearRegressionGD(learning_rate=0.099, epochs=2000, auto=False)
    line_pipe = Pipeline([
    ("prep", linear_preprocessor),
    ("model", linear_model_stacking)
    ])

    rf_pipe = Pipeline([
        ("prep", rf_preprocessor),
        ("model", rf_model)
    ])

    #Đã tìm hyperparameters tốt cho base model KNN trong stacking
    knn_model_stacking = KNNRegressorCustom(weights='distance', k=30)
    knn_pipe = Pipeline([
        ("prep", knn_preprocessor),
        ("model", knn_model_stacking)
    ])

    base_models = [line_pipe, knn_pipe, rf_pipe]
    #Đã tìm hyperparameters tốt cho meta-model Ridge Regression (alpha = 10)
    meta_model = RidgeRegressionMetaModel(alpha=10.0, fit_intercept=True, auto=False)
    
    stacking_model = StackingEnsembleCustom(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=5
    )
    
    stacking_model.fit(X_train_df, y_train)
    y_pred = stacking_model.predict(X_test_df)
    
    print(f"\nSaving models to '{models_dir}/'...")

    with open(f'{models_dir}/stacking_custom.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)

    # Save evaluation metrics
    metrics_stacking = RegressionMetrics(y_test, y_pred).to_dict()
    with open(f"{models_dir}/stacking_metrics.json", "w") as f:
        json.dump(metrics_stacking, f, indent=4)

    print("Models saved successfully!")
    
    #Lưu kết quả dự đoán và kết quả thật để so sánh
    with open(f'{models_dir}/stacking_custom.pkl', 'rb') as f:
        stacking_model = pickle.load(f)
    with open(f'{models_dir}/linear_custom.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    with open(f'{models_dir}/knn_custom.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open(f'{models_dir}/rf_custom.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    y_pred_loaded = stacking_model.predict(X_test_df)
    df_check = pd.DataFrame({
        "y_test": y_test.flatten(),
        "y_stacking": y_pred_loaded.flatten(),
        "y_linear": linear_model.predict(linear_preprocessor.transform(X_test_df)).flatten(),
        "y_knn": knn_model.predict(knn_preprocessor.transform(X_test_df)).flatten(),
        "y_rf": rf_model.predict(rf_preprocessor.transform(X_test_df)).flatten()
    })
    print(df_check.head(10))
    df_check.to_csv("data/proccesed/model_check.csv", index=False)
if __name__ == "__main__":
    main()