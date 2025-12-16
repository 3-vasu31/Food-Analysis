import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_and_clean_data

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("Dataset Overview")
st.write("Quick summary of the Zomato restaurant dataset")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'zomato.csv', 'zomato.csv')
    return load_and_clean_data(data_path)

try:
    with st.spinner('Loading dataset...'):
        df = load_data()
    
    st.success(f"Successfully loaded {len(df):,} restaurants")
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Unique Locations", df['location'].nunique())
    with col3:
        st.metric("Cuisine Types", df['cuisines'].nunique())
    with col4:
        st.metric("Avg Rating", f"{df['clean_rate'].mean():.2f}/5.0")
    with col5:
        st.metric("Avg Cost (₹)", f"₹{df['approx_cost(for two people)'].mean():.0f}")
    
    st.divider()
    
    # Dataset Info
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📋 Column Information")
        
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
        st.write(f"**Total Columns:** {len(df.columns)}")
        
    
    with col_right:
        st.subheader("🔢 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.divider()
    
    # Categorical columns summary
    st.subheader("📑 Categorical Features Distribution")
    
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        st.write("**Location Distribution (Top 10)**")
        location_counts = df['location'].value_counts().head(10)
        st.bar_chart(location_counts)
        
        st.write("**Online Order Availability**")
        st.bar_chart(df['online_order'].value_counts())
    
    with cat_col2:
        st.write("**Cuisine Distribution (Top 10)**")
        cuisine_counts = df['cuisines'].value_counts().head(10)
        st.bar_chart(cuisine_counts)
        
        st.write("**Table Booking Availability**")
        st.bar_chart(df['book_table'].value_counts())
    
    st.divider()
    
    # Sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.sample(20), use_container_width=True)
    
    # Download option
    st.subheader("💾 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="zomato_cleaned.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please ensure the data file is located at: data/zomato.csv/zomato.csv")
