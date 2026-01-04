import streamlit as st
import numpy as np
import pandas as pd
import pickle, json
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Thêm src vào path để import custom models
sys.path.append('src')

# Import custom models (nếu cần tạo mới)
from data.preprocess_for_model.knn import KNN_FEATURES
from data.preprocess_for_model.random_forest import NUMERIC_FEATURES
from data.preprocess_for_model.linear import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from models.base_model.linear_regression import LinearRegressionGD
from models.base_model.random_forest import RandomForestRegressor
from models.base_model.knn import KNNRegressorCustom
from models.stacking import ManualStackingEnsemble as StackingEnsembleCustom
from models.meta_model.ridge_regression import RidgeRegressionMetaModel
from data.preprocess_for_model import linear_preprocessor
from data.preprocess_for_model import knn_preprocessor
from data.preprocess_for_model import rf_preprocessor
models_dir = 'models_trained'
@st.cache_resource
def load_and_fit_preprocessors():
    """Load training data và fit preprocessors"""
    
    print("Loading training data to fit preprocessors...")
    
    try:
        # Load training data (giống như khi train models)
        train_df = pd.read_csv("data/split/train.csv")
        print(f"Loaded training data: {train_df.shape}")
        
        # Tách features và target
        X_train = train_df.drop(columns=["rating"])
        y_train = train_df["rating"]
        
        # 1. Fit linear preprocessor
        print("Fitting linear preprocessor...")
        X_linear_processed = linear_preprocessor.fit_transform(X_train)
        print(f"Linear preprocessor fitted. Output shape: {X_linear_processed.shape}")
        
        # 2. Fit KNN preprocessor
        print("Fitting KNN preprocessor...")
        knn_preprocessor.fit_transform(X_train)
        print(f"KNN preprocessor fitted.")
        
        # 3. Fit Random Forest preprocessor
        print("Fitting Random Forest preprocessor...")
        rf_preprocessor.fit_transform(X_train)
        print(f"Random Forest preprocessor fitted.")

        return {
            'linear_preprocessor': linear_preprocessor,
            'knn_preprocessor': knn_preprocessor,
            'rf_preprocessor': rf_preprocessor,
            'X_train_sample': X_train.iloc[0:1]  # Lấy mẫu để biết cấu trúc
        }
        
    except Exception as e:
        print(f"Error fitting preprocessors: {e}")
        return None
@st.cache_resource
def load_custom_models():
    """Load các custom models đã train"""
    models = {}
    
    try:
        # Load từ file pickle
        with open(f'{models_dir}/linear_custom.pkl', 'rb') as f:
            models['linear'] = pickle.load(f)
        
        with open(f'{models_dir}/rf_custom.pkl', 'rb') as f:
            models['random_forest'] = pickle.load(f)
        
        with open(f'{models_dir}/knn_custom.pkl', 'rb') as f:
            models['knn'] = pickle.load(f)
        
        with open(f'{models_dir}/stacking_custom.pkl', 'rb') as f:
            models['stacking'] = pickle.load(f)
        
        st.success("Loaded custom models successfully!")
        
    except FileNotFoundError:
        st.warning("Models not found. Creating dummy custom models...")
        
        #Tạo dummy custom models
        models['linear'] = LinearRegressionGD(alpha=1.0)
        models['random_forest'] = RandomForestRegressor(n_estimators=10)
        models['knn'] = KNNRegressorCustom(k=5)
        
        # Dummy stacking
        base_models = [models['linear'], models['random_forest'], models['knn']]
        meta_model = RidgeRegressionMetaModel(alpha=0.5, fit_intercept=True)
        models['stacking'] = StackingEnsembleCustom(
            base_models=base_models,
            meta_model=meta_model,
            n_folds=3
        )
        
        # Train dummy models với data giả
        np.random.seed(42)
        X_dummy = np.random.randn(100, 5)
        y_dummy = X_dummy[:, 0] * 2 + np.random.randn(100) * 0.5
        
        for model in models.values():
            if hasattr(model, 'fit'):
                model.fit(X_dummy, y_dummy)
    
    return models

st.cache_resource
def load_metrics_models():

    with open(f"{models_dir}/linear_metrics.json", "r") as f:
        linear_metrics = json.load(f)
        
    with open(f"{models_dir}/rf_metrics.json", "r") as f:
        rf_metrics = json.load(f)
        
    with open(f"{models_dir}/knn_metrics.json", "r") as f:
        knn_metrics = json.load(f)
        
    with open(f"{models_dir}/stacking_metrics.json", "r") as f:
        stacking_metrics = json.load(f)
    
    st.success("Loaded metrics models successfully!")
    
    metrics_df = pd.DataFrame([{
        "Model": "Linear Regression",
        **linear_metrics
    }, {
        "Model": "Random Forest",
        **rf_metrics
    }, {
        "Model": "KNN",
        **knn_metrics
    }, {
        "Model": "Stacking Ensemble",
        **stacking_metrics
    }])
    return metrics_df

@st.cache_data
def load_sample_data():
    """Load sample data từ file CSV"""
    try:
        df = pd.read_csv('data/processed/clean_movies_data_1975_2025.csv')
        data = pd.read_csv('data/processed/model_check.csv')
        return df, data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def display_model_info(model, model_name):
    """Hiển thị thông tin về custom model"""
    
    st.markdown(f"### {model_name} Details")
    
    if model_name == "Linear Regression":
        if hasattr(model, 'get_params'):
            params = model.get_params()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Learning Rate", f"{params.get('LR', 'N/A')}")
                st.metric("Weights Shape", f"{params.get('weights', 'N/A').shape}")
            with col2:
                st.metric("Epochs", str(params.get('epochs', 'N/A')))
                st.metric("Bias", str(round(params.get('bias', 'N/A'), 4)) if params.get('bias', None) is not None else 'N/A')
    
    elif model_name == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            if hasattr(model, 'trees'):
                st.metric("Number of Trees", len(model.trees))
            st.metric("Max Depth", getattr(model, 'max_depth', 'N/A'))
        with col2:
            st.metric("Estimators", getattr(model, 'n_estimators', 'N/A'))
            st.metric("Random State", getattr(model, 'random_state', 'N/A'))
    
    elif model_name == "KNN Regressor":
        params = model.get_params()
        col1, col2 = st.columns(2)

        with col1:
            st.metric("k (Neighbors)", "min(20, sqrt(n_sample))" if params.get("k") == 'auto' else params.get("k", "N/A"))
            st.metric("Best k", params.get("best_k", "N/A"))

        with col2:
            st.metric("Weights", params.get("weights", "N/A"))
            st.metric("Metric", params.get("metric", "N/A"))

        if hasattr(model, "X_train"):
            st.metric("Training Samples", len(model.X_train))
            
    elif model_name == "Stacking Ensemble":
        metrics_df = load_metrics_models()
        df_performance = metrics_df.copy()
        st.dataframe(df_performance)
        r = df_performance.loc[df_performance['Model'] == 'Stacking Ensemble', 'R²'].values
        summary = model.get_stacking_summary(r[0] if len(r) > 0 else 0.0)

        st.title("Stacking Ensemble Summary")
        st.subheader("Ensemble Performance")
        st.metric(
            label="R² Score (Final Stacking Model)",
            value=f"{summary['ensemble']['r2_score']:.4f}"
        )

        st.subheader("Meta-model Configuration")
        st.json(summary["meta_model"])
        
        st.subheader("Meta-model Weights")
        df_weights = pd.DataFrame.from_dict(
            summary["weights"],
            orient="index",
            columns=["Normalized Weight"]
        )
        df_weights.index.name = "Base Model"
        st.dataframe(df_weights, use_container_width=True)

        st.subheader("Final Prediction Formula")
        st.code(summary["ensemble"]["formula"], language="text")

        st.info(
            "Meta-model weights giải thích cách các base models kết hợp. "
            "Đánh giá cuối cùng sẽ dựa trên ensemble"
        )

def convert_duration_to_minutes(duration_str):
    try:
        if isinstance(duration_str, str):
            if 'h' in duration_str and 'm' in duration_str:
                hours = int(duration_str.split('h')[0].strip())
                minutes = int(duration_str.split('h')[1].split('m')[0].strip())
                return hours * 60 + minutes
            elif 'h' in duration_str:
                return int(duration_str.split('h')[0].strip()) * 60
            elif 'm' in duration_str:
                return int(duration_str.split('m')[0].strip())
        return int(duration_str)
    except:
        return 120  # default

def preprocess_user_input(user_input_dict, feature_type='linear', preprocessors_info=None):
    features_df = pd.DataFrame([user_input_dict])
    
    print(f"Raw user input: {user_input_dict}")
    print(f"DataFrame columns: {features_df.columns.tolist()}")
    
    if feature_type == 'linear':
        try:
            # Transform
            features_processed = preprocessors_info['linear_preprocessor'].transform(features_df)
            print(f"Processed shape: {features_processed.shape}")
            
            return features_processed
            
        except Exception as e:
            print(f"Linear preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    elif feature_type == 'knn':
        try:

            features_processed = preprocessors_info['knn_preprocessor'].transform(features_df)
            print(f" Processed shape: {features_processed.shape}")
            
            return features_processed
            
        except Exception as e:
            print(f"KNN preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    elif feature_type == 'rf':
        try:
            features_processed = preprocessors_info['rf_preprocessor'].transform(features_df)
            print(f"RF features processed, shape: {features_processed.shape}")
            return features_processed
            
        except Exception as e:
            print(f"RF preprocessing error: {e}")
            # Fallback
            return features_df.values
    else:
        return features_df.values
    
def make_predictions(models, user_input_dict, preprocessors_info):   
    predictions = {}
    
    # 1. LINEAR REGRESSION
    print("\n1. Linear Regression Prediction:")
    try:
        # Preprocess cho Linear
        linear_features = preprocess_user_input(user_input_dict, feature_type='linear', preprocessors_info=preprocessors_info)
        linear_pred = models['linear'].predict(linear_features)
        # Lấy giá trị prediction
        if isinstance(linear_pred, np.ndarray):
            if linear_pred.ndim > 1:
                pred_value = float(linear_pred[0][0])
            else:
                pred_value = float(linear_pred[0])
        else:
            pred_value = float(linear_pred)
        
        predictions['Linear Regression'] = pred_value
        print(f"Prediction: {pred_value:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        predictions['Linear Regression'] = 6.5
    
    # 2. RANDOM FOREST
    print("\n2. Random Forest Prediction:")
    try:
        rf_features = preprocess_user_input(user_input_dict, feature_type='rf', preprocessors_info=preprocessors_info)
        rf_pred = models['random_forest'].predict(rf_features)
        
        predictions['Random Forest'] = float(rf_pred[0])
        print(f"Prediction: {predictions['Random Forest']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        predictions['Random Forest'] = 6.5
    
    # 3. KNN
    print("\n3. KNN Prediction:")
    try:
        knn_features = preprocess_user_input(user_input_dict, feature_type='knn', preprocessors_info=preprocessors_info)
        knn_pred = models['knn'].predict(knn_features)
        
        predictions['KNN Regressor'] = float(knn_pred[0])
        print(f"Prediction: {predictions['KNN Regressor']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        predictions['KNN Regressor'] = 6.5
    
    # 4. STACKING ENSEMBLE
    print("\n4. Stacking Ensemble Prediction:")
    try:

        features_df = pd.DataFrame([user_input_dict])
        stacking_columns = [
            'budget', 'votes', 'duration', 'meta_score', 'grossworldwide',
            'genres', 'countries_origin', 'languages', 'mpa', 'year'
        ]
        
        for col in stacking_columns:
            if col not in features_df.columns:
                if col in ['budget', 'votes', 'grossworldwide']:
                    features_df[col] = 0
                elif col in ['duration', 'meta_score', 'year']:
                    features_df[col] = 0
                else:
                    features_df[col] = ""
        # Predict với stacking pipeline
        stacking_pred = models['stacking'].predict(features_df[stacking_columns])
        
        if isinstance(stacking_pred, (np.ndarray, list)) and len(stacking_pred) > 0:
            predictions['Stacking Ensemble'] = float(stacking_pred[0])
        else:
            predictions['Stacking Ensemble'] = float(stacking_pred)
        
        print(f"Prediction: {predictions['Stacking Ensemble']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        if predictions:
            weights = {'Linear Regression': 0.3, 'Random Forest': 0.4, 'KNN Regressor': 0.3}
            weighted_sum = 0
            total_weight = 0
            
            for model_name, weight in weights.items():
                if model_name in predictions:
                    weighted_sum += predictions[model_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                predictions['Stacking Ensemble'] = weighted_sum / total_weight
            else:
                predictions['Stacking Ensemble'] = np.mean(list(predictions.values()))
        else:
            predictions['Stacking Ensemble'] = 6.5
    
    for model_name, pred in predictions.items():
        print(f"{model_name:<20}: {pred:.4f}")
    
    return predictions

def display_evaluation_charts():
    
    st.subheader("Model Performance Evaluation")
    
    metrics_df = load_metrics_models()
    df_performance = metrics_df.copy()
    st.dataframe(df_performance)

    # Biểu đồ 1: MAE và RMSE
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        x=df_performance['Model'],
        y=df_performance['MAE'],
        name='MAE',
        marker_color='#FF6B6B'
    ))
    
    fig1.add_trace(go.Bar(
        x=df_performance['Model'],
        y=df_performance['RMSE'],
        name='RMSE',
        marker_color='#4ECDC4'
    ))
    
    fig1.update_layout(
        title='Model Error Metrics (Lower is Better)',
        barmode='group',
        yaxis_title='Error Value',
        height=400
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Biểu đồ 2: R² Score
    fig2 = go.Figure(go.Bar(
        x=df_performance['Model'],
        y=df_performance['R²'],
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        text=df_performance['R²'].round(3),
        textposition='outside'
    ))
    
    fig2.update_layout(
        title='R² Score (Higher is Better)',
        yaxis_title='R² Score',
        yaxis_range=[0, 1],
        height=350
    )
    
    st.plotly_chart(fig2, use_container_width=True)
      
    # Hiển thị bảng performance
    st.subheader("Performance Metrics Table")
    st.dataframe(df_performance.style.highlight_max(subset=['R²'], color='lightgreen')
                              .highlight_min(subset=['MAE', 'RMSE'], color='lightcoral'),
                 use_container_width=True)

def main():
    st.set_page_config(
        page_title="Movie Rating Predictor - Custom ML",
        page_icon="🎬",
        layout="wide"
    )
    if 'preprocessors_info' not in st.session_state:
        with st.spinner("Fitting preprocessors with training data..."):
            preprocessors_info = load_and_fit_preprocessors()
            st.session_state.preprocessors_info = preprocessors_info
    
    preprocessors_info = st.session_state.preprocessors_info
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎬 Movie Rating Predictor with Custom ML Models</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"><b>All models implemented from scratch! 🛠️</b> | Predict IMDb ratings based on movie features</p>', unsafe_allow_html=True)
    
    # Load models và data
    with st.spinner("Loading custom models and data..."):
        models = load_custom_models()
        df_sample, data_sample = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        app_mode = st.radio(
            "Select Mode:",
            ["Predict New Movie", "Model Evaluation", "View Dataset", "Model Details"]
        )
        
        st.markdown("---")
        st.header("Sample Data Info")
        
        if not df_sample.empty:
            st.metric("Total Movies", len(df_sample))
            st.metric("Years Range", f"{df_sample['year'].min()} - {df_sample['year'].max()}")
            
            avg_rating = df_sample['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        else:
            st.warning("No sample data loaded")
        
        st.markdown("---")
        st.caption("Using Custom ML Models")
    
    # Main content
    if app_mode == "Predict New Movie":
        st.header("Predict Movie Rating")
        
        # Tạo form với tất cả features
        with st.form("movie_prediction_form"):
            st.subheader("Movie Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                movie_title = st.text_input("Movie Title*", "The Matrix")
                year = st.number_input("Release Year*", 1900, 2024, 1999)
                # Duration nhập theo format "2h 30m"
                duration_hours = st.number_input("Hours", 0, 5, 2)
                duration_minutes = st.number_input("Minutes", 0, 59, 16)
                duration_input = f"{duration_hours}h {duration_minutes}m"
                
                mpa_rating = st.selectbox(
                    "MPA Rating*",
                    ["PG", "R", "PG-13", "G", "NC-17", "Not Rated", "Unknown", "Approved"],
                    index=1
                )
                
                meta_score = st.slider("Meta Score (0-100)", 0, 100, 73)
                
            with col2:
                # Multi-select cho genres
                available_genres = [
                    "Action", "Adventure", "Animation", "Comedy", "Crime",
                    "Drama", "Fantasy", "Horror", "Mystery", "Romance",
                    "Sci-Fi", "Thriller", "Documentary", "Family", "Musical",
                    "Western", "Biography", "History", "Sport", "War",
                    "Music", "Film-Noir", "Reality-TV", "Talk-Show", "News"
                ]
                selected_genres = st.multiselect(
                    "Genres* (select up to 5)",
                    available_genres,
                    ["Action", "Sci-Fi"],
                    max_selections=5
                )
                genres = str(selected_genres) if selected_genres else "['Action']"
                
                countries = st.text_input("Countries of Origin* (comma separated)", "United States, Australia")
                languages = st.text_input("Languages* (comma separated)", "English")
                
            with col3:
                # Numerical inputs
                votes = st.number_input("Number of Votes*", 0, 10000000, 1000000, 
                                    help="Approximate number of IMDb votes")
                
                # Financials
                budget = st.number_input("Budget (USD)*", 1000, 500000000, 63000000, step=1000000,
                                        help="Production budget in USD")
                opening_gross = st.number_input("Opening Weekend Gross (USD)", 0, 500000000, 27872000, step=1000000)
                worldwide_gross = st.number_input("Worldwide Gross (USD)", 0, 3000000000, 463517383, step=1000000)
                us_canada_gross = st.number_input("US/Canada Gross (USD)", 0, 2000000000, 171479930, step=1000000)
            
            # Additional features từ dataset
            st.subheader("Additional Details")
            
            col4, col5 = st.columns(2)
            
            with col4:
                # Production company
                production_company = st.text_input("Production Company", "Warner Bros., Village Roadshow Pictures")
                
                # Filming locations
                filming_locations = st.text_input("Filming Locations", "Sydney, Australia; Chicago, USA")
                
                # Awards (simplified)
                awards_wins = st.number_input("Awards Wins", 0, 100, 4)
                awards_nominations = st.number_input("Awards Nominations", 0, 100, 0)
                
            with col5:
                # Director và stars (simplified)
                director_name = st.text_input("Main Director", "Lana Wachowski, Lilly Wachowski")
                writer_name = st.text_input("Main Writer", "Lana Wachowski, Lilly Wachowski")
                
                # Star power (0-1 scale)
                star_power = st.slider("Star Power (0-1)", 0.0, 1.0, 0.8, 0.1,
                                    help="Overall popularity of main cast (0=unknown, 1=A-list)")
                
                # Director experience (years)
                director_exp = st.slider("Director Experience (years)", 0, 50, 5)
            
            # Nút submit
            submitted = st.form_submit_button(
                "Predict Rating with Custom Models", 
                type="primary", 
                use_container_width=True
            )
        
        # Khi user submit form
        if submitted:
            # Validate required fields
            required_fields = {
                "Movie Title": movie_title,
                "Genres": selected_genres,
                "Countries": countries,
                "Languages": languages
            }
            
            missing_fields = [field for field, value in required_fields.items() if not value]
            
            if missing_fields:
                st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
            else:
                # Chuẩn bị input data với TẤT CẢ features từ dataset
                user_input = {
                    # Basic info
                    'title': movie_title,
                    'year': year,
                    'duration': convert_duration_to_minutes(duration_input),
                    'mpa': mpa_rating,
                    'genres': genres,
                    'countries_origin': f"['{countries}']" if ',' not in countries else str([c.strip() for c in countries.split(',')]),
                    'languages': f"['{languages}']" if ',' not in languages else str([l.strip() for l in languages.split(',')]),
                    
                    # Ratings và votes
                    'votes': votes,
                    'meta_score': meta_score,
                    
                    # Financials
                    'budget': budget,
                    'opening_weekend_gross': opening_gross,
                    'grossworldwide': worldwide_gross,
                    'gross_us_canada': us_canada_gross,
                    
                    # Additional features
                    'production_company': production_company,
                    'filming_locations': filming_locations,
                    
                    # Derived features (sẽ được tính tự động)
                    'director_exp': director_exp,
                    'star_power': star_power,
                    
                    # Awards
                    'awards_content': f"Won {awards_wins} awards. {awards_nominations} nominations." if awards_wins > 0 or awards_nominations > 0 else "No information",
                    
                    # Crew (simplified)
                    'directors': f"['{director_name}']",
                    'writers': f"['{writer_name}']",
                    'stars': "['Keanu Reeves', 'Laurence Fishburne', 'Carrie-Anne Moss']"  # Default
                }
                
                # Thêm các log features (tự động tính)
                user_input['budget_log'] = np.log1p(budget)
                user_input['votes_log'] = np.log1p(votes)
                user_input['grossworldwide_log'] = np.log1p(worldwide_gross)
                user_input['gross_us_canada_log'] = np.log1p(us_canada_gross)
                user_input['opening_weekend_gross_log'] = np.log1p(opening_gross)
                
                # Thêm count features
                user_input['genres_count'] = len(selected_genres)
                user_input['countries_count'] = len(countries.split(',')) if ',' in countries else 1
                user_input['languages_count'] = len(languages.split(',')) if ',' in languages else 1
                
                # Thêm features cho KNN
                user_input['num_stars'] = 3  # Default
                user_input['num_awards'] = awards_wins + awards_nominations
                
                # Fix typo từ dataset (grossworldwwide có 2 'w')
                user_input['grossworldwwide'] = worldwide_gross
                user_input['grossworldwwide_log'] = np.log1p(worldwide_gross)
                
                # Preprocess và predict
                with st.spinner("Xử lý features và tiến hành dự đoán..."):
                    try:
                        predictions = make_predictions(models, user_input, preprocessors_info)
                        
                        # Hiển thị kết quả
                        st.success("Hoàn thành dự đoán!")
                        
                        # Hiển thị input summary
                        with st.expander("Tổng quan Input"):
                            input_summary = {
                                'Basic Info': {
                                    'Title': movie_title,
                                    'Year': year,
                                    'Duration': f"{duration_input} ({user_input['duration']} minutes)",
                                    'MPA Rating': mpa_rating,
                                    'Genres': selected_genres,
                                    'Countries': countries,
                                    'Languages': languages
                                },
                                'Ratings': {
                                    'Votes': f"{votes:,}",
                                    'Meta Score': meta_score
                                },
                                'Financials (USD)': {
                                    'Budget': f"${budget:,}",
                                    'Opening Weekend': f"${opening_gross:,}",
                                    'Worldwide Gross': f"${worldwide_gross:,}",
                                    'US/Canada Gross': f"${us_canada_gross:,}"
                                }
                            }
                            
                            for category, items in input_summary.items():
                                st.markdown(f"**{category}**")
                                for key, value in items.items():
                                    st.text(f"  {key}: {value}")
                        
                        # Main prediction card
                        main_pred = predictions.get('Stacking Ensemble', 6.5)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    border-radius: 15px; color: white; margin: 1rem 0;">
                            <h2>{movie_title} ({year})</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;">{main_pred:.2f}/10</h1>
                            <p>Predicted IMDb Rating</p>
                            <p style="font-size: 0.9rem; opacity: 0.9;">Based on {len(predictions)} custom ML models</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model comparisons
                        st.subheader("Model Comparison")
                        
                        cols = st.columns(4)
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                        model_names = list(predictions.keys())
                        
                        for idx in range(min(4, len(model_names))):
                            with cols[idx]:
                                model_name = model_names[idx]
                                rating = predictions[model_name]
                                diff = rating - main_pred
                                
                                # Tên ngắn gọn
                                short_name = model_name.replace(' (Custom)', '').replace(' Ensemble', '')
                                
                                st.markdown(f"""
                                <div style="padding: 1rem; background-color: {colors[idx]}20; 
                                            border-radius: 10px; border-left: 5px solid {colors[idx]};">
                                    <h4 style="color: {colors[idx]}; margin: 0;">{short_name}</h4>
                                    <h3 style="margin: 0.5rem 0;">{rating:.2f}</h3>
                                    <small style="color: {'#4CAF50' if diff > 0 else '#F44336' if diff < 0 else '#666'};">
                                        {f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"}
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Bar chart visualization
                        st.subheader("Predictions Visualization")
                        
                        fig_pred = go.Figure(data=[
                            go.Bar(
                                x=list(predictions.keys()),
                                y=list(predictions.values()),
                                marker_color=colors[:len(predictions)],
                                text=[f'{v:.2f}' for v in predictions.values()],
                                textposition='outside',
                                textfont=dict(size=14)
                            )
                        ])
                        
                        fig_pred.update_layout(
                            title="Predictions from Custom Models",
                            yaxis_title="Predicted Rating",
                            yaxis_range=[0, 10],
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Feature impact explanation
                        st.subheader("Các features quan trọng")
                        
                        # Giải thích các feature quan trọng
                        feature_impact = {
                            'Budget': f"${budget/1000000:.1f}M",
                            'Meta Score': f"{meta_score}/100",
                            'Votes': f"{votes/1000000:.1f}M",
                            'Duration': f"{user_input['duration']} min",
                            'Genres': f"{len(selected_genres)} genres",
                            'Director Experience': f"{director_exp} years",
                            'Star Power': f"{star_power:.1f}/1.0"
                        }
                        
                        # Hiển thị dạng metrics
                        impact_cols = st.columns(len(feature_impact))
                        
                        for idx, (feature, value) in enumerate(feature_impact.items()):
                            with impact_cols[idx]:
                                st.metric(feature, value)
                        
                        # Recommendations
                        st.subheader("Gợi ý để cải thiện rating")
                        
                        recommendations = []
                        
                        if budget < 50000000:
                            recommendations.append("Tăng ngân sách sản xuất để đạt được giá trị sản xuất tốt hơn.")
                        
                        if meta_score < 60:
                            recommendations.append("Tập trung vào đánh giá của giới phê bình.")
                        
                        if star_power < 0.7:
                            recommendations.append("Hãy cân nhắc lựa chọn những diễn viên đã có tên tuổi.")
                        
                        if len(selected_genres) < 2:
                            recommendations.append("Thêm các thể loại khác để tiếp cận nhiều khán giả hơn.")
                        
                        if recommendations:
                            for rec in recommendations:
                                st.info(f"• {rec}")
                        else:
                            st.success("Tất cả các chỉ số đều cho thấy triển vọng tốt, hứa hẹn xếp hạng cao!")
                        
                        # Save prediction history
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        prediction_record = {
                            'title': movie_title,
                            'year': year,
                            'predictions': predictions,
                            'timestamp': pd.Timestamp.now(),
                            'input_features': {
                                k: v for k, v in user_input.items() 
                                if k not in ['title', 'directors', 'writers', 'stars', 'awards_content']
                            }
                        }
                        
                        st.session_state.prediction_history.append(prediction_record)
                        
                        # Show history
                        with st.expander("Prediction History"):
                            if len(st.session_state.prediction_history) > 0:
                                history_df = pd.DataFrame([
                                    {
                                        'Title': record['title'],
                                        'Year': record['year'],
                                        'Stacking': record['predictions'].get('Stacking Ensemble', 'N/A'),
                                        'Time': record['timestamp'].strftime('%H:%M:%S')
                                    }
                                    for record in st.session_state.prediction_history[-5:]  # Last 5
                                ])
                                st.dataframe(history_df, use_container_width=True)
                            else:
                                st.write("No prediction history yet.")
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.info("Try adjusting the input values or check if all required fields are filled correctly.")
                        
                        # Debug info
                        with st.expander("🛠️ Debug Information"):
                            st.write("User Input Keys:", list(user_input.keys()))
                            st.write("User Input Sample:", {k: user_input[k] for k in list(user_input.keys())[:5]})
                            
                            if preprocessors_info:
                                st.write("Preprocessor expects:", 
                                        f"{len(preprocessors_info.get('linear_columns', []))} columns")
    
    elif app_mode == "Model Evaluation":
        st.header("Model Performance Evaluation")
        
        # Hiển thị biểu đồ đánh giá
        display_evaluation_charts()
        
        # Thêm phần giải thích
        st.markdown("---")
        st.subheader("Giải thích Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **MAE (Mean Absolute Error):**
            - Sai số tuyệt đối trung bình giữa giá trị dự đoán và giá trị thực tế
            - Lower is better
            - Dễ hiểu
            
            **RMSE (Root Mean Square Error):**
            - Căn bậc hai của bình phương trung bình
            - Phạt nặng hơn đối với các lỗi lớn.
            - Lower is better
            """)
        
        with col2:
            st.markdown("""
            **R² Score (Coefficient of Determination):**
            - Tỷ lệ phương sai được giải thích bởi mô hình
            - Ranges from 0 to 1
            - Higher is better
            - 1 = dự đoán hoàn hảo
            """)
    
    elif app_mode == "View Dataset":
        st.header("Movie Dataset Overview")
        
        if not df_sample.empty:
            # Hiển thị dataset
            st.dataframe(df_sample, use_container_width=True)
            
            # Thống kê cơ bản
            st.subheader("Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Movies", len(df_sample))
                st.metric("Years Range", f"{df_sample['year'].min()} - {df_sample['year'].max()}")
            
            with col2:
                avg_rating = df_sample['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}")
                st.metric("Highest Rating", f"{df_sample['rating'].max():.1f}")
            
            with col3:
                total_budget = df_sample['budget'].sum() / 1_000_000
                st.metric("Total Budget (M)", f"${total_budget:.0f}M")
                st.metric("Languages", df_sample['languages'].nunique())
            
            # Biểu đồ phân phối ratings
            st.subheader("Rating Distribution")
            
            fig_dist = px.histogram(
                df_sample,
                x='rating',
                nbins=20,
                title='Distribution of Movie Ratings',
                labels={'rating': 'IMDb Rating'},
                color_discrete_sequence=['#667eea']
            )
            
            fig_dist.update_layout(
                bargap=0.1,
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Biểu đồ budget vs rating
            st.subheader("Budget vs Rating Analysis")
            
            fig_scatter = px.scatter(
                df_sample,
                x='budget',
                y='rating',
                size='votes',
                color='mpa',
                hover_name='title',
                title='Budget vs Rating with MPA Rating',
                labels={'budget': 'Budget (USD)', 'rating': 'IMDb Rating'},
                size_max=30
            )
            upper = df_sample['budget'].quantile(0.95)

            fig_scatter.update_layout(
                xaxis=dict(
                    range=[0, upper],
                    autorange=False
                )
            )   
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Hiển thị top movies
            st.subheader("Top Rated Movies")
            
            top_movies = df_sample.nlargest(5, 'rating')[['title', 'year', 'rating', 'genres']]
            st.dataframe(top_movies.style.highlight_max(subset=['rating'], color='lightgreen'),
                        use_container_width=True)
        else:
            st.warning("No dataset available")
        if not data_sample.empty:
            st.subheader("Predicted Ratings Sample")
            df = data_sample.copy()
            st.dataframe(data_sample, use_container_width=True)
            fig = go.Figure()

            # Đường chéo lý tưởng (y = x)
            fig.add_trace(go.Scatter(
                x=[df['y_test'].min(), df['y_test'].max()],
                y=[df['y_test'].min(), df['y_test'].max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Lý tưởng (y = x)',
                showlegend=True
            ))

            # Thêm dữ liệu từng mô hình
            models = ['y_stacking', 'y_linear', 'y_knn', 'y_rf']
            colors = ['blue', 'green', 'orange', 'red']
            names = ['Stacking', 'Linear', 'KNN', 'Random Forest']

            for i, (model, color, name) in enumerate(zip(models, colors, names)):
                fig.add_trace(go.Scattergl(
                    x=df['y_test'],
                    y=df[model],
                    mode='markers',
                    marker=dict(size=5, opacity=0.5),
                    name=name
                ))

            # Cập nhật layout
            fig.update_layout(
                title='So sánh dự đoán của các mô hình với giá trị thực tế',
                xaxis_title='Giá trị thực tế (y_test)',
                yaxis_title='Giá trị dự đoán',
                template='plotly_white',
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                width=1000,
                height=600
            )

            # Hiển thị biểu đồ
            st.plotly_chart(fig, use_container_width=True)

    
    else:  # Model Details
        st.header("🔧 Custom Model Details")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏗️ Ridge Regression", 
            "🌲 Random Forest", 
            "📏 KNN", 
            "🏗️ Stacking Ensemble"
        ])
        
        with tab1:
            display_model_info(models['linear'], "Linear Regression")
            st.markdown("""
            **Tổng quan thuật toán:**
            - Hồi quy tuyến tính với Giảm dần độ dốc (Gradient Descent)
            - Cập nhật trọng số lặp đi lặp lại để tối thiểu hóa hàm mất mát
            - Công thức cập nhật: w = w - α * ∇J(w)

            **Ưu điểm:**
            - Dễ triển khai và hiểu
            - Hiệu quả với tập dữ liệu lớn
            - Có thể áp dụng cho các mô hình phức tạp hơn
            - Là nền tảng cho nhiều thuật toán ML

            **Quy trình huấn luyện:**
            1. Khởi tạo trọng số ngẫu nhiên
            2. Tính dự đoán và sai số
            3. Tính gradient của hàm mất mát
            4. Cập nhật trọng số theo learning rate
            5. Lặp lại cho đến khi hội tụ
            """)
        
        with tab2:
            display_model_info(models['random_forest'], "Random Forest")
            st.markdown("""
            **Tổng quan thuật toán:**
            - Tập hợp nhiều cây quyết định (ensemble learning)
            - Bootstrap aggregating (bagging) với tính ngẫu nhiên kép
            - Mỗi cây huấn luyện trên tập con dữ liệu và đặc trưng

            **Ưu điểm:**
            - Độ chính xác cao, giảm overfitting
            - Xử lý tốt quan hệ phi tuyến
            - Mạnh mẽ với outliers và nhiễu
            - Cung cấp độ quan trọng đặc trưng

            **Quy trình huấn luyện:**
            1. Tạo n tập con dữ liệu bằng bootstrap sampling
            2. Huấn luyện cây quyết định trên mỗi tập con
            3. Tại mỗi node, chọn ngẫu nhiên tập con đặc trưng
            4. Tìm split point tốt nhất trong tập con đặc trưng
            5. Tổng hợp kết quả bằng trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại)
            """)
        
        with tab3:
            display_model_info(models['knn'], "KNN Regressor")
            st.markdown("""
            **Tổng quan thuật toán:**
            - Học dựa trên thể hiện (instance-based learning)
            - Tìm k điểm dữ liệu gần nhất trong không gian đặc trưng
            - Dự đoán bằng trung bình có trọng số của k láng giềng
            - Khoảng cách phổ biến: Euclidean, Manhattan
            - Chọn k theo phương pháp sqrt(n) hoặc cross-validation

            **Ưu điểm:**
            - Đơn giản và trực quan, dễ triển khai
            - Không cần giai đoạn huấn luyện (lazy learning)
            - Thích ứng nhanh với dữ liệu mới
            - Phi tham số, không cần giả định phân phối

            **Quy trình huấn luyện:**
            1. Lưu trữ toàn bộ tập dữ liệu huấn luyện
            2. Với mỗi điểm cần dự đoán, tính khoảng cách đến tất cả điểm huấn luyện (euclidean, manhattan)
            3. Sắp xếp khoảng cách và chọn k điểm gần nhất
            4. Tính dự đoán bằng trung bình (uniform) hoặc trung bình có trọng số theo khoảng cách (distance)
            """)
        
        with tab4:
            display_model_info(models['stacking'], "Stacking Ensemble")
            st.markdown("### 🏗️ Stacking Ensemble Architecture")
            
            st.markdown("""
            **Kiến trúc:**
            ```
            Level 0 (Base Models):
            ├── Ridge Regression (Base_Model 1)
            ├── K-Nearest Neighbors (Base_Model 2)
            └── Random Forest (Base_Model 3)

            Level 1 (Meta Model):
            └── Ridge Regression (Meta_Model)
            ```
            
            **Quy trình huấn luyện:**
            1. Huấn luyện các mô hình cơ sở trên dữ liệu huấn luyện
            2. Tạo dự đoán cross-validated (siêu đặc trưng)
            3. Huấn luyện siêu mô hình trên siêu đặc trưng
            4. Kết hợp dự đoán tối ưu

            **Ưu điểm:**
            - Tận dụng điểm mạnh của nhiều mô hình
            - Giảm thiên lệch và phương sai
            - Thường đạt hiệu suất tốt nhất
            """)
            
            # Visualization
            st.image("https://images.viblo.asia/b83e6f12-31e1-41c5-9414-ef1ef6307dd5.png", 
                    caption="Stacking Ensemble Architecture", use_column_width=True)

if __name__ == "__main__":
    main()