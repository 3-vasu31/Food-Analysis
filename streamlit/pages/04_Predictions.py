import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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


st.title("🔮 ML Predictions")
st.write("Select a model to train, compare results, and predict restaurant ratings or prices.")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data', 'zomato.csv', 'zomato.csv'
    )
    return load_and_clean_data(data_path, for_modeling=True)


# ---------------------------------------------------------------------------
# Feature Preparation with One-Hot Encoding
# ---------------------------------------------------------------------------
@st.cache_data
def prepare_features(df, target='rating'):
    """Prepare features using One-Hot Encoding for categorical variables."""
    df_model = df.copy()

    # Binary features
    df_model['online_order_yes'] = (df_model['online_order'] == 'Yes').astype(int)
    df_model['book_table_yes'] = (df_model['book_table'] == 'Yes').astype(int)

    # Categorical features to one-hot encode
    cat_features = ['location', 'cuisines', 'rest_type', 'listed_in(type)']

    # One-Hot Encode each categorical column
    ohe_frames = []
    ohe_columns_map = {}
    for col in cat_features:
        dummies = pd.get_dummies(df_model[col], prefix=col, drop_first=True).astype(int)
        ohe_frames.append(dummies)
        ohe_columns_map[col] = list(dummies.columns)

    ohe_df = pd.concat(ohe_frames, axis=1)

    # Build feature matrix
    binary_cols = ['online_order_yes', 'book_table_yes']

    if target == 'rating':
        numeric_cols = ['approx_cost(for two people)']
        y = df_model['clean_rate']
    else:
        numeric_cols = ['clean_rate']
        y = df_model['approx_cost(for two people)']

    X = pd.concat([df_model[binary_cols + numeric_cols], ohe_df], axis=1)
    feature_names = list(X.columns)

    return X, y, feature_names, ohe_columns_map, cat_features


# ---------------------------------------------------------------------------
# Accuracy within tolerance
# ---------------------------------------------------------------------------
def accuracy_within_tolerance(y_true, y_pred, tolerance):
    """Percentage of predictions within ±tolerance of the actual value."""
    return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100


# ---------------------------------------------------------------------------
# Available model definitions
# ---------------------------------------------------------------------------
def get_available_models():
    """Return dict of model name -> description."""
    models = {'Random Forest': 'A robust tree-based ensemble — good all-rounder.'}
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = 'Gradient boosting — often the most accurate.'
    else:
        models['Random Forest (Tuned)'] = 'A tuned Random Forest variant (XGBoost not installed).'
    models['Neural Network'] = 'A deep neural network (MLP) — captures complex patterns.'
    return models


def _create_model(name):
    """Instantiate an untrained model by name."""
    if name == 'Random Forest':
        return RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42
        )
    elif name == 'XGBoost':
        return XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    elif name == 'Random Forest (Tuned)':
        return RandomForestRegressor(
            n_estimators=300, max_depth=25, min_samples_leaf=1, random_state=42
        )
    elif name == 'Neural Network':
        return MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64, 32),
            activation='relu', solver='adam', max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.1
        )


# ---------------------------------------------------------------------------
# Prepare data splits (cached so it is only computed once per target)
# ---------------------------------------------------------------------------
@st.cache_resource
def prepare_splits(_df, target='rating'):
    """Return train/test splits, scaler, and metadata — computed once."""
    X, y, feature_names, ohe_columns_map, cat_features = prepare_features(_df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tolerance = 0.5 if target == 'rating' else 200

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            scaler, feature_names, ohe_columns_map,
            cat_features, list(X.columns), tolerance)


# ---------------------------------------------------------------------------
# Train a single model (cached per model+target — instant on re-select)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_single_model(_df, model_name, target='rating'):
    """Train one model and return (model, metrics dict)."""
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled,
     _scaler, _fn, _ohe, _cf, _ac, tolerance) = prepare_splits(_df, target)

    model = _create_model(model_name)
    is_nn = 'Neural' in model_name
    Xtr = X_train_scaled if is_nn else X_train
    Xte = X_test_scaled if is_nn else X_test

    model.fit(Xtr, y_train)
    y_pred_train = model.predict(Xtr)
    y_pred_test = model.predict(Xte)

    metrics = {
        'R² Score (Train)': r2_score(y_train, y_pred_train),
        'R² Score (Test)': r2_score(y_test, y_pred_test),
        'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
        'MAE (Test)': mean_absolute_error(y_test, y_pred_test),
        'Accuracy % (Train)': accuracy_within_tolerance(y_train.values, y_pred_train, tolerance),
        'Accuracy % (Test)': accuracy_within_tolerance(y_test.values, y_pred_test, tolerance),
        'y_test': y_test,
        'y_pred': y_pred_test,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Plot Helpers
# ---------------------------------------------------------------------------
def plot_model_comparison(results, metric_key, title, ylabel):
    """Bar chart comparing all models on a given metric."""
    names = list(results.keys())
    train_key = f'{metric_key} (Train)'
    test_key = f'{metric_key} (Test)'
    train_vals = [results[n][train_key] for n in names]
    test_vals = [results[n][test_key] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, train_vals, width, label='Train', color='#667eea', alpha=0.85)
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test', color='#f76c6c', alpha=0.85)

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name, target_label):
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=12, edgecolors='k', linewidths=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(f'Actual {target_label}', fontweight='bold')
    ax.set_ylabel(f'Predicted {target_label}', fontweight='bold')
    ax.set_title(f'Actual vs Predicted — {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def interpret_model(name, metrics, target, tolerance):
    """Return a list of plain-English interpretations for a model's metrics."""
    lines = []

    # R² interpretation
    r2 = metrics['R² Score (Test)']
    pct = r2 * 100
    if r2 >= 0.85:
        lines.append(f"**{name}** explains **{pct:.1f}%** of the variation — excellent fit.")
    elif r2 >= 0.70:
        lines.append(f"**{name}** explains **{pct:.1f}%** of the variation — good fit.")
    elif r2 >= 0.50:
        lines.append(f"**{name}** explains **{pct:.1f}%** of the variation — moderate fit, some patterns are missed.")
    else:
        lines.append(f"**{name}** explains only **{pct:.1f}%** of the variation — weak fit.")

    # MAE interpretation
    mae = metrics['MAE (Test)']
    if target == 'rating':
        lines.append(f"On average, predictions are off by **{mae:.2f} stars**.")
    else:
        lines.append(f"On average, predictions are off by **₹{mae:.0f}**.")

    # Accuracy % interpretation
    acc = metrics['Accuracy % (Test)']
    if target == 'rating':
        lines.append(f"**{acc:.1f}%** of predictions land within **±{tolerance} stars** of the actual rating.")
    else:
        lines.append(f"**{acc:.1f}%** of predictions land within **±₹{tolerance}** of the actual price.")

    # Overfitting check
    r2_gap = metrics['R² Score (Train)'] - metrics['R² Score (Test)']
    if r2_gap > 0.10:
        lines.append(f"⚠️ Train R² is {r2_gap:.2f} higher than Test R² — signs of overfitting.")
    else:
        lines.append(f"✅ Train and Test R² are close (gap: {r2_gap:.2f}) — no overfitting.")

    return lines


# ---------------------------------------------------------------------------
# Build a single-row input vector for prediction (one-hot encoded)
# ---------------------------------------------------------------------------
def build_prediction_input(all_columns, ohe_columns_map, cat_values, binary_values, numeric_value):
    row = pd.DataFrame(np.zeros((1, len(all_columns))), columns=all_columns)

    for col, val in binary_values.items():
        if col in row.columns:
            row[col] = val

    ohe_col_set = {k for v in ohe_columns_map.values() for k in v}
    numeric_col = [c for c in all_columns
                   if c not in ohe_col_set and c not in binary_values][0]
    row[numeric_col] = numeric_value

    for orig_col, selected_val in cat_values.items():
        ohe_col_name = f"{orig_col}_{selected_val}"
        if ohe_col_name in row.columns:
            row[ohe_col_name] = 1

    return row


# ===================================================================
# MAIN APP
# ===================================================================
try:
    df = load_data()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("🎛️ Configuration")
    prediction_type = st.sidebar.selectbox(
        "Prediction Target",
        ["Restaurant Rating", "Price for Two People"]
    )
    target_key = 'rating' if prediction_type == "Restaurant Rating" else 'price'

    if not XGBOOST_AVAILABLE:
        st.sidebar.warning("XGBoost not installed — `pip install xgboost`")

    # Pre-compute data splits (cached — runs only once per target)
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler, feature_names, ohe_columns_map,
     cat_features, all_columns, tolerance) = prepare_splits(df, target=target_key)

    available = get_available_models()
    model_list = list(available.keys())

    # Initialise session-state cache for trained models
    if 'trained_cache' not in st.session_state:
        st.session_state.trained_cache = {}   # key = (model_name, target_key)

    # ------------------------------------------------------------------
    # 1. Select & Train a Model
    # ------------------------------------------------------------------
    st.header("🤖 Select & Train a Model")

    selected_model = st.selectbox(
        "Choose a model to train",
        model_list,
        format_func=lambda n: f"{n}  —  {available[n]}"
    )

    col_train, col_all = st.columns([1, 1])
    train_one = col_train.button(f"🚀 Train {selected_model}", type="primary", use_container_width=True)
    train_all = col_all.button("📊 Train All & Compare", use_container_width=True)

    models_to_train = []
    if train_one:
        models_to_train = [selected_model]
    if train_all:
        models_to_train = model_list

    # Train requested models (cached ones return instantly)
    for name in models_to_train:
        cache_key = (name, target_key)
        if cache_key not in st.session_state.trained_cache:
            with st.spinner(f"Training {name} …"):
                model, metrics = train_single_model(df, name, target=target_key)
                st.session_state.trained_cache[cache_key] = (model, metrics)

    # Gather all trained results for the current target
    trained_models = {}
    results = {}
    for name in model_list:
        cache_key = (name, target_key)
        if cache_key in st.session_state.trained_cache:
            model, metrics = st.session_state.trained_cache[cache_key]
            trained_models[name] = model
            results[name] = metrics

    if not results:
        st.info("👆 Select a model and click **Train** to get started. Already-trained models will load instantly.")
        st.stop()

    # Show which models are ready
    status_cols = st.columns(len(model_list))
    for i, name in enumerate(model_list):
        with status_cols[i]:
            if name in trained_models:
                st.success(f"✅ {name}")
            else:
                st.warning(f"⏳ {name}")

    # ---- Metrics explanation ----
    st.markdown(f"""
    **How to read the metrics:**
    | Metric | What it means |
    |---|---|
    | **R² Score** | How much of the variation the model explains (1.0 = perfect, 0 = no better than guessing the average) |
    | **MAE** | Average prediction error in {'stars' if target_key == 'rating' else '₹'} — lower is better |
    | **Accuracy %** | % of predictions within ±{tolerance} {'stars' if target_key == 'rating' else '₹'} of the actual value — higher is better |
    """)

    # ---- Summary table ----
    summary_rows = []
    for name, m in results.items():
        summary_rows.append({
            'Model': name,
            'R² Score': round(m['R² Score (Test)'], 3),
            'MAE': round(m['MAE (Test)'], 3),
            'Accuracy %': round(m['Accuracy % (Test)'], 1),
        })
    summary_df = pd.DataFrame(summary_rows)

    best_model_name = summary_df.loc[summary_df['R² Score'].idxmax(), 'Model']
    st.dataframe(
        summary_df.style.highlight_max(subset=['R² Score', 'Accuracy %'], color='#d4edda')
                        .highlight_min(subset=['MAE'], color='#d4edda'),
        use_container_width=True, hide_index=True
    )
    if len(results) > 1:
        best_r2 = results[best_model_name]['R² Score (Test)']
        st.info(f"🏆 **Best model**: {best_model_name} (R² = {best_r2:.3f})")

    # ---- Comparison bar charts (only when >1 model trained) ----
    if len(results) > 1:
        st.subheader("📈 Metric Comparison")
        tab_r2, tab_mae, tab_acc = st.tabs(["R² Score", "MAE", "Accuracy %"])

        with tab_r2:
            fig = plot_model_comparison(results, 'R² Score', 'R² Score — Higher is Better', 'R² Score')
            st.pyplot(fig)
        with tab_mae:
            ylabel = 'MAE (₹)' if target_key == 'price' else 'MAE (Stars)'
            fig = plot_model_comparison(results, 'MAE', 'Mean Absolute Error — Lower is Better', ylabel)
            st.pyplot(fig)
        with tab_acc:
            tol_label = f'±₹{tolerance}' if target_key == 'price' else f'±{tolerance} stars'
            fig = plot_model_comparison(results, 'Accuracy %',
                                        f'Predictions within {tol_label} — Higher is Better',
                                        'Accuracy %')
            st.pyplot(fig)

    st.divider()

    # ------------------------------------------------------------------
    # 2. What the model(s) tell us (interpretation)
    # ------------------------------------------------------------------
    st.header("💬 What Do the Models Tell Us?")

    for name in results:
        is_best = len(results) > 1 and name == best_model_name
        with st.expander(f"{'🏆 ' if is_best else ''}{name}", expanded=(name == best_model_name)):
            for line in interpret_model(name, results[name], target_key, tolerance):
                st.markdown(f"- {line}")

    st.divider()

    # ------------------------------------------------------------------
    # 3. Actual vs Predicted
    # ------------------------------------------------------------------
    show_model = best_model_name if len(results) > 1 else list(results.keys())[0]
    st.header(f"📊 Actual vs Predicted — {show_model}")
    target_label = 'Rating (stars)' if target_key == 'rating' else 'Cost for Two (₹)'
    fig_avp = plot_actual_vs_predicted(
        results[show_model]['y_test'].values,
        results[show_model]['y_pred'],
        show_model, target_label
    )
    st.pyplot(fig_avp)
    st.caption("Points close to the red dashed line are accurate predictions.")

    st.divider()

    # ------------------------------------------------------------------
    # 4. Feature Importance (tree-based models only)
    # ------------------------------------------------------------------
    st.header("🌳 Feature Importance")

    tree_models = {n: m for n, m in trained_models.items()
                   if hasattr(m, 'feature_importances_')}

    if tree_models:
        selected_imp_model = st.selectbox("Select model", list(tree_models.keys()))
        importances = tree_models[selected_imp_model].feature_importances_

        # Aggregate OHE columns back to the original feature name
        original_importance = {}
        for idx, col_name in enumerate(feature_names):
            matched = False
            for orig_col, ohe_cols in ohe_columns_map.items():
                if col_name in ohe_cols:
                    original_importance[orig_col] = original_importance.get(orig_col, 0) + importances[idx]
                    matched = True
                    break
            if not matched:
                original_importance[col_name] = original_importance.get(col_name, 0) + importances[idx]

        imp_df = (pd.DataFrame(list(original_importance.items()),
                               columns=['Feature', 'Importance'])
                  .sort_values('Importance', ascending=False))

        fig_imp, ax_imp = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.5)))
        colors = plt.cm.viridis(imp_df['Importance'] / imp_df['Importance'].max())
        ax_imp.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='black')
        ax_imp.set_xlabel('Importance', fontweight='bold')
        ax_imp.set_title(f'Feature Importance — {selected_imp_model}', fontweight='bold')
        ax_imp.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_imp)

        with st.expander("📋 Importance Values"):
            st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)
            top_feature = imp_df.iloc[0]['Feature']
            top_pct = imp_df.iloc[0]['Importance']
            st.write(f"**Most influential feature**: {top_feature} ({top_pct:.1%})")
    else:
        st.info("Feature importance is only available for tree-based models.")

    st.divider()

    # ------------------------------------------------------------------
    # 5. Make a Prediction
    # ------------------------------------------------------------------
    st.header("🎯 Make a Prediction")

    pred_model_name = st.selectbox(
        "Choose model for prediction",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name) if best_model_name in trained_models else 0
    )

    col1, col2 = st.columns(2)

    cat_inputs = {}
    with col1:
        for feat in cat_features:
            label = feat.replace('_', ' ').replace('(', '').replace(')', '').title()
            cat_inputs[feat] = st.selectbox(label, sorted(df[feat].unique()), key=f"pred_{feat}")
    with col2:
        online_order = st.selectbox("Online Order", ["Yes", "No"], key="pred_online")
        book_table = st.selectbox("Table Booking", ["Yes", "No"], key="pred_book")
        if target_key == 'rating':
            numeric_val = st.number_input("Approx Cost for Two (₹)",
                                          min_value=0, max_value=6000, value=500, step=50)
        else:
            numeric_val = st.slider("Restaurant Rating", 1.0, 5.0, 3.5, 0.1)

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        binary_vals = {
            'online_order_yes': 1 if online_order == "Yes" else 0,
            'book_table_yes': 1 if book_table == "Yes" else 0,
        }

        input_row = build_prediction_input(
            all_columns, ohe_columns_map, cat_inputs, binary_vals, numeric_val
        )

        chosen_model = trained_models[pred_model_name]
        if 'Neural' in pred_model_name:
            input_arr = scaler.transform(input_row)
        else:
            input_arr = input_row

        prediction = chosen_model.predict(input_arr)[0]
        mae = results[pred_model_name]['MAE (Test)']

        if target_key == 'rating':
            prediction = np.clip(prediction, 1.0, 5.0)

            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #667eea;">Predicted Rating: {prediction:.2f} / 5.0</h2>
                <p style="text-align: center;">± {mae:.2f} stars (average error)</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.subheader("What does this mean?")
            if prediction >= 4.5:
                st.success(
                    "⭐⭐⭐⭐⭐ **Excellent!** This restaurant is predicted to be among the top-rated. "
                    "Expect outstanding food quality, great service, and a wonderful dining experience."
                )
            elif prediction >= 3.8:
                st.info(
                    "⭐⭐⭐⭐ **Very Good!** This restaurant is predicted to have strong ratings. "
                    "Most diners would find the food and experience satisfying."
                )
            elif prediction >= 3.0:
                st.info(
                    "⭐⭐⭐ **Average.** This restaurant is predicted to have a decent but not standout rating. "
                    "The food and service are likely acceptable, with room for improvement."
                )
            elif prediction >= 2.0:
                st.warning(
                    "⭐⭐ **Below Average.** This restaurant may have mixed reviews. "
                    "Some diners could be disappointed with food quality or service."
                )
            else:
                st.error(
                    "⭐ **Poor.** This restaurant is predicted to have low ratings. "
                    "Significant issues with food, service, or hygiene are likely reported by diners."
                )
        else:
            prediction = max(0, prediction)

            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #667eea;">Predicted Cost for Two: ₹{prediction:.0f}</h2>
                <p style="text-align: center;">± ₹{mae:.0f} (average error)</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.subheader("What does this mean?")
            if prediction >= 3500:
                st.info(
                    "💎 **Luxury Dining.** Expect a premium fine-dining experience with high-end ingredients, "
                    "elaborate plating, and top-notch service. Think special occasions and celebrations."
                )
            elif prediction >= 1500:
                st.info(
                    "🍽️ **Premium Casual.** A mid-to-high range restaurant offering quality food in a "
                    "comfortable setting. Great for a nice dinner out without breaking the bank."
                )
            elif prediction >= 500:
                st.info(
                    "🍴 **Mid-Range.** Affordable and satisfying — this is a typical sit-down restaurant. "
                    "Good value for money with a variety of dishes."
                )
            elif prediction >= 200:
                st.info(
                    "🥘 **Budget-Friendly.** An economical choice — think casual eateries, street food joints, "
                    "or quick-service restaurants. Filling meals at wallet-friendly prices."
                )
            else:
                st.info(
                    "🍜 **Very Economical.** Extremely affordable — likely a small eatery or street food stall "
                    "offering basic but tasty meals at rock-bottom prices."
                )

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
