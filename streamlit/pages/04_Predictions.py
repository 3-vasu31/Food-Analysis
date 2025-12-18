import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("LightGBM not installed. Install with: pip install lightgbm")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_and_clean_data

st.set_page_config(page_title="ML Predictions", page_icon="🔮", layout="wide")

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.prediction-box {
    background: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
}
</style>
""", unsafe_allow_html=True)


st.title("🔮 Advanced ML Predictions")
st.write("Compare multiple ML models with custom hyperparameters for rating and price prediction")

# Load model configurations
@st.cache_data
def load_model_config():
    """Load model configurations from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration if file doesn't exist
        return {
            "random_forest": {
                "display_name": "Random Forest",
                "configs": {
                    "Default": {"n_estimators": 100, "max_depth": 15, "random_state": 42},
                    "Shallow": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
                    "Deep": {"n_estimators": 200, "max_depth": 25, "random_state": 42}
                }
            },
            "gradient_boosting": {
                "display_name": "Gradient Boosting",
                "configs": {
                    "Default": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
                    "Aggressive": {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 5, "random_state": 42}
                }
            }
        }

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'zomato.csv', 'zomato.csv')
    return load_and_clean_data(data_path, for_modeling=True)

def get_model_instance(model_type, config_params):
    """Get model instance based on type and parameters"""
    if model_type == "random_forest":
        return RandomForestRegressor(**config_params)
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(**config_params)
    elif model_type == "xgboost" and XGBOOST_AVAILABLE:
        return XGBRegressor(**config_params)
    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        return LGBMRegressor(**config_params)
    else:
        return RandomForestRegressor(**config_params)

@st.cache_resource
def train_model(df, target='rating', model_type='random_forest', config_params=None):
    """Train selected model for rating or price prediction"""
    
    # Prepare data
    df_model = df.copy()
    
    # Encode categorical variables
    le_location = LabelEncoder()
    le_cuisines = LabelEncoder()
    le_rest_type = LabelEncoder()
    le_listed = LabelEncoder()
    
    df_model['location_encoded'] = le_location.fit_transform(df_model['location'])
    df_model['cuisines_encoded'] = le_cuisines.fit_transform(df_model['cuisines'])
    df_model['rest_type_encoded'] = le_rest_type.fit_transform(df_model['rest_type'])
    df_model['listed_in_encoded'] = le_listed.fit_transform(df_model['listed_in(type)'])
    df_model['online_order_encoded'] = (df_model['online_order'] == 'Yes').astype(int)
    df_model['book_table_encoded'] = (df_model['book_table'] == 'Yes').astype(int)
    
    # Select features
    if target == 'rating':
        features = ['location_encoded', 'cuisines_encoded', 'rest_type_encoded', 
                   'listed_in_encoded', 'online_order_encoded', 'book_table_encoded',
                   'approx_cost(for two people)']
        X = df_model[features]
        y = df_model['clean_rate']
    else:  # price
        features = ['location_encoded', 'cuisines_encoded', 'rest_type_encoded',
                   'listed_in_encoded', 'online_order_encoded', 'book_table_encoded',
                   'clean_rate']
        X = df_model[features]
        y = df_model['approx_cost(for two people)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get model instance
    if config_params is None:
        config_params = {"n_estimators": 100, "random_state": 42}
    model = get_model_instance(model_type, config_params)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Calculate residuals
    residuals = y_test - y_pred_test
    
    encoders = {
        'location': le_location,
        'cuisines': le_cuisines,
        'rest_type': le_rest_type,
        'listed_in': le_listed
    }
    
    metrics = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'residuals': residuals
    }
    
    return model, encoders, metrics, features

def plot_error_analysis(y_test, y_pred, residuals, target_name):
    """Plot comprehensive error analysis visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q Plot (Residuals normality check)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def interpret_errors(residuals, metrics, target):
    """Provide interpretation of model errors"""
    interpretations = []
    
    # Overall performance
    if metrics['Test R²'] >= 0.75:
        interpretations.append("**Excellent Model Performance**: R² score indicates the model explains >80% of variance.")
    elif metrics['Test R²'] >= 0.6:
        interpretations.append("**Good Model Performance**: R² score shows decent predictive power.")
    else:
        interpretations.append("**Moderate Performance**: Model may benefit from feature engineering or different algorithms.")
    
    # Overfitting check
    r2_diff = abs(metrics['Train R²'] - metrics['Test R²'])
    if r2_diff > 0.1:
        interpretations.append(f"**Possible Overfitting**: Train R² ({metrics['Train R²']:.3f}) significantly higher than Test R² ({metrics['Test R²']:.3f})")
    else:
        interpretations.append(f"**No Overfitting Detected**: Train and Test R² are similar (diff: {r2_diff:.3f})")
    
    # Residual analysis
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    if abs(mean_residual) < 0.1:
        interpretations.append(f"**Unbiased Predictions**: Mean residual ≈ 0 ({mean_residual:.4f})")
    else:
        if mean_residual > 0:
            interpretations.append(f"**Systematic Underestimation**: Model tends to predict lower than actual (bias: {mean_residual:.4f})")
        else:
            interpretations.append(f"**Systematic Overestimation**: Model tends to predict higher than actual (bias: {mean_residual:.4f})")
    
    # Error magnitude
    if target == 'rating':
        if metrics['Test MAE'] < 0.3:
            interpretations.append(f"**Low Prediction Error**: Average error of {metrics['Test MAE']:.3f} stars")
        elif metrics['Test MAE'] < 0.5:
            interpretations.append(f"**Acceptable Error**: Average error of {metrics['Test MAE']:.3f} stars")
        else:
            interpretations.append(f"**High Error**: Average error of {metrics['Test MAE']:.3f} stars may impact reliability")
    else:  # price
        if metrics['Test MAE'] < 100:
            interpretations.append(f"**Low Price Error**: Average error of ₹{metrics['Test MAE']:.0f}")
        elif metrics['Test MAE'] < 200:
            interpretations.append(f"**Acceptable Price Error**: Average error of ₹{metrics['Test MAE']:.0f}")
        else:
            interpretations.append(f"**High Price Error**: Average error of ₹{metrics['Test MAE']:.0f}")
    
    return interpretations

try:
    df = load_data()
    model_configs = load_model_config()
    
    # Sidebar for model selection
    st.sidebar.header("🎛️ Model Configuration")
    
    # Choose prediction type
    prediction_type = st.sidebar.selectbox("Prediction Type", ["Restaurant Rating", "Price for Two People"])
    
    # Model selection
    available_models = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting"
    }
    
    if XGBOOST_AVAILABLE:
        available_models["XGBoost"] = "xgboost"
    if LIGHTGBM_AVAILABLE:
        available_models["LightGBM"] = "lightgbm"
    
    selected_model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
    selected_model_type = available_models[selected_model_name]
    
    # Get available configurations for selected model
    if selected_model_type in model_configs:
        config_names = list(model_configs[selected_model_type]['configs'].keys())
        selected_config = st.sidebar.selectbox("Configuration", config_names)
        config_params = model_configs[selected_model_type]['configs'][selected_config]
    else:
        selected_config = "Default"
        config_params = {"n_estimators": 100, "random_state": 42}
    
    # Display selected configuration
    st.sidebar.markdown("### 📋 Hyperparameters")
    st.sidebar.json(config_params)
    
    st.divider()
    
    if prediction_type == "Restaurant Rating":
        st.subheader(f"🌟 Rating Prediction - {selected_model_name}")
        st.write(f"Using **{selected_config}** configuration")
        
        # Train model
        with st.spinner(f"Training {selected_model_name} model..."):
            model, encoders, metrics, features = train_model(df, target='rating', 
                                                             model_type=selected_model_type, 
                                                             config_params=config_params)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test R²", f"{metrics['Test R²']:.4f}", 
                     delta=f"{(metrics['Test R²'] - metrics['Train R²']):.4f}")
        with col2:
            st.metric("Test RMSE", f"{metrics['Test RMSE']:.4f}")
        with col3:
            st.metric("Test MAE", f"{metrics['Test MAE']:.4f}")
        with col4:
            overfitting_score = abs(metrics['Train R²'] - metrics['Test R²'])
            st.metric("Overfit Check", f"{overfitting_score:.4f}", 
                     delta="Low" if overfitting_score < 0.1 else "High")
        
        # Detailed metrics
        with st.expander("Detailed Metrics Comparison"):
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Training Set': [f"{metrics['Train R²']:.4f}", 
                               f"{metrics['Train RMSE']:.4f}",
                               f"{metrics['Train MAE']:.4f}"],
                'Test Set': [f"{metrics['Test R²']:.4f}",
                           f"{metrics['Test RMSE']:.4f}",
                           f"{metrics['Test MAE']:.4f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
        
        st.divider()
        
        # Error Analysis
        st.subheader("📈 Error Analysis & Diagnostics")
        
        tab1, tab2 = st.tabs(["📉 Visualization", "💬 Interpretation"])
        
        with tab1:
            error_fig = plot_error_analysis(metrics['y_test'], metrics['y_pred'], 
                                           metrics['residuals'], 'Rating')
            st.pyplot(error_fig)
            
            st.write("""
            **Understanding the plots:**
            - **Actual vs Predicted**: Points should be close to the red diagonal line
            - **Residual Plot**: Should show random scatter around zero (no patterns)
            - **Distribution**: Should be bell-shaped and centered at zero
            - **Q-Q Plot**: Points should follow the diagonal for normal distribution
            """)
        
        with tab2:
            interpretations = interpret_errors(metrics['residuals'], metrics, 'rating')
            for interpretation in interpretations:
                st.markdown(interpretation)
        
        st.divider()
        
        # Input form for prediction
        st.subheader("Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.selectbox("Location", sorted(df['location'].unique()))
            cuisines = st.selectbox("Cuisine Type", sorted(df['cuisines'].unique()))
            rest_type = st.selectbox("Restaurant Type", sorted(df['rest_type'].unique()))
        
        with col2:
            listed_in = st.selectbox("Listed In", sorted(df['listed_in(type)'].unique()))
            online_order = st.selectbox("Online Order", ["Yes", "No"])
            book_table = st.selectbox("Table Booking", ["Yes", "No"])
            cost = st.number_input("Approximate Cost for Two (₹)", min_value=0, max_value=5000, value=500, step=50)
        
        if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
            # Encode inputs
            location_encoded = encoders['location'].transform([location])[0]
            cuisines_encoded = encoders['cuisines'].transform([cuisines])[0]
            rest_type_encoded = encoders['rest_type'].transform([rest_type])[0]
            listed_in_encoded = encoders['listed_in'].transform([listed_in])[0]
            online_order_encoded = 1 if online_order == "Yes" else 0
            book_table_encoded = 1 if book_table == "Yes" else 0
            
            # Prepare input
            input_data = np.array([[location_encoded, cuisines_encoded, rest_type_encoded,
                                   listed_in_encoded, online_order_encoded, book_table_encoded, cost]])
            
            # Predict
            prediction = model.predict(input_data)[0]
            prediction = np.clip(prediction, 1.0, 5.0)
            
            # Calculate confidence based on MAE
            confidence_range = metrics['Test MAE']
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #667eea;">Predicted Rating: {prediction:.2f} / 5.0</h2>
                <p style="text-align: center;">Confidence Interval: {max(1.0, prediction - confidence_range):.2f} - {min(5.0, prediction + confidence_range):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")  # Spacing
            
            # Interpretation
            if prediction >= 4.5:
                st.success("⭐⭐⭐⭐⭐ **Excellent!** This restaurant is likely to be highly rated.")
            elif prediction >= 3.5:
                st.info("⭐⭐⭐⭐ **Very Good!** This restaurant should have good ratings.")
            elif prediction >= 3.0:
                st.info("⭐⭐⭐ **Good!** This restaurant has average to above-average ratings.")
            else:
                st.warning("⭐⭐ **Below Average.** This restaurant might have lower ratings.")
    
    else:  # Price prediction
        st.subheader(f"💰 Price Prediction - {selected_model_name}")
        st.write(f"Using **{selected_config}** configuration")
        
        # Train model
        with st.spinner(f"Training {selected_model_name} model..."):
            model, encoders, metrics, features = train_model(df, target='price',
                                                             model_type=selected_model_type,
                                                             config_params=config_params)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test R²", f"{metrics['Test R²']:.4f}",
                     delta=f"{(metrics['Test R²'] - metrics['Train R²']):.4f}")
        with col2:
            st.metric("Test RMSE", f"₹{metrics['Test RMSE']:.0f}")
        with col3:
            st.metric("Test MAE", f"₹{metrics['Test MAE']:.0f}")
        with col4:
            overfitting_score = abs(metrics['Train R²'] - metrics['Test R²'])
            st.metric("Overfit Check", f"{overfitting_score:.4f}",
                     delta="Low" if overfitting_score < 0.1 else "High")
        
        # Detailed metrics
        with st.expander("Detailed Metrics Comparison"):
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Training Set': [f"{metrics['Train R²']:.4f}",
                               f"₹{metrics['Train RMSE']:.0f}",
                               f"₹{metrics['Train MAE']:.0f}"],
                'Test Set': [f"{metrics['Test R²']:.4f}",
                           f"₹{metrics['Test RMSE']:.0f}",
                           f"₹{metrics['Test MAE']:.0f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
        
        st.divider()
        
        # Error Analysis
        st.subheader("📈 Error Analysis & Diagnostics")
        
        tab1, tab2 = st.tabs(["📉 Visualization", "💬 Interpretation"])
        
        with tab1:
            error_fig = plot_error_analysis(metrics['y_test'], metrics['y_pred'],
                                           metrics['residuals'], 'Price')
            st.pyplot(error_fig)
            
            st.write("""
            **Understanding the plots:**
            - **Actual vs Predicted**: Points should be close to the red diagonal line
            - **Residual Plot**: Should show random scatter around zero (no patterns)
            - **Distribution**: Should be bell-shaped and centered at zero
            - **Q-Q Plot**: Points should follow the diagonal for normal distribution
            """)
        
        with tab2:
            interpretations = interpret_errors(metrics['residuals'], metrics, 'price')
            for interpretation in interpretations:
                st.markdown(interpretation)
        
        st.divider()
        
        # Input form
        st.subheader("🎯 Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.selectbox("Location", sorted(df['location'].unique()))
            cuisines = st.selectbox("Cuisine Type", sorted(df['cuisines'].unique()))
            rest_type = st.selectbox("Restaurant Type", sorted(df['rest_type'].unique()))
        
        with col2:
            listed_in = st.selectbox("Listed In", sorted(df['listed_in(type)'].unique()))
            online_order = st.selectbox("Online Order", ["Yes", "No"])
            book_table = st.selectbox("Table Booking", ["Yes", "No"])
            rating = st.slider("Restaurant Rating", 1.0, 5.0, 3.5, 0.1)
        
        if st.button("💰 Predict Price", type="primary", use_container_width=True):
            # Encode inputs
            location_encoded = encoders['location'].transform([location])[0]
            cuisines_encoded = encoders['cuisines'].transform([cuisines])[0]
            rest_type_encoded = encoders['rest_type'].transform([rest_type])[0]
            listed_in_encoded = encoders['listed_in'].transform([listed_in])[0]
            online_order_encoded = 1 if online_order == "Yes" else 0
            book_table_encoded = 1 if book_table == "Yes" else 0
            
            # Prepare input
            input_data = np.array([[location_encoded, cuisines_encoded, rest_type_encoded,
                                   listed_in_encoded, online_order_encoded, book_table_encoded, rating]])
            
            # Predict
            prediction = model.predict(input_data)[0]
            prediction = max(0, prediction)
            
            # Calculate confidence based on MAE
            confidence_range = metrics['Test MAE']
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #667eea;">Predicted Cost for Two: ₹{prediction:.0f}</h2>
                <p style="text-align: center;">Confidence Interval: ₹{max(0, prediction - confidence_range):.0f} - ₹{prediction + confidence_range:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")  # Spacing
            
            # Price interpretation
            if prediction >= 1500:
                st.info("💎 **Luxury Dining** - Premium restaurant with high-end pricing")
            elif prediction >= 1000:
                st.info("🍽️ **Premium** - Mid to high-range restaurant")
            elif prediction >= 450:
                st.info("🍴 **Mid-Range** - Affordable dining option")
            else:
                st.info("🥘 **Budget-Friendly** - Economical dining choice")
    
    st.divider()
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    
    feature_names = ['Location', 'Cuisines', 'Restaurant Type', 'Listed In', 'Online Order', 'Table Booking']
    if prediction_type == "Restaurant Rating":
        feature_names.append('Cost')
    else:
        feature_names.append('Rating')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max())
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("📋 Feature Importance Values"):
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
        
        # Interpretation
        top_feature = importance_df.iloc[0]['Feature']
        top_importance = importance_df.iloc[0]['Importance']
        st.write(f"""
        **Key Insights:**
        - **Most Important Feature**: {top_feature} ({top_importance:.1%} importance)
        - Features are ranked by their contribution to the model's predictions
        - Higher importance = stronger influence on {prediction_type.lower()}
        """)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
