import streamlit as st
import pandas as pd
import os
import sys

# Configure page
st.set_page_config(
    page_title="Food Delivery Analysis",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 200rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<p class="main-header">🍕 Food Delivery Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze restaurant data, discover insights, and predict ratings & prices</p>', unsafe_allow_html=True)

st.write("""
### Welcome to the Food Delivery Analysis Platform!

This application provides comprehensive analysis of restaurant data from Zomato.

### What you can do:

- **Overview**: Get a quick summary of the dataset with key statistics
- **Data Analysis**: Explore detailed visualizations and trends
- **Insights**: Discover patterns in delivery zones, cuisines, and restaurant types
- **Predictions**: Predict restaurant ratings and prices

### Get Started:

Use the sidebar navigation to explore different sections.
""")

st.divider()

# Quick stats
col1, col2, col3, col4 = st.columns(4)

try:
    from utils import load_and_clean_data
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zomato.csv', 'zomato.csv')
    
    if os.path.exists(data_path):
        with st.spinner('Loading data...'):
            df = load_and_clean_data(data_path)
            
        with col1:
            st.metric("Total Restaurants", f"{len(df):,}")
        with col2:
            st.metric("Locations", f"{df['location'].nunique()}")
        with col3:
            avg_rating = df['clean_rate'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}/5.0")
        with col4:
            avg_cost = df['approx_cost(for two people)'].mean()
            st.metric("Avg Cost for Two", f"₹{avg_cost:.0f}")
            
        st.success("Data loaded successfully! Navigate to other pages using the sidebar.")
except Exception as e:
    st.info("ℹ️ Navigate to other pages using the sidebar to start exploring.")

st.divider()
st.caption("Made with ❤️ using Streamlit | Data Source: Zomato")
