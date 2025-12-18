import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent dircectory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_and_clean_data

st.set_page_config(page_title="Data Analysis", page_icon="📈", layout="wide")

st.title("Data Analysis & Visualization")
st.write("Explore detailed visualizations and trends in the restaurant data")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'zomato.csv', 'zomato.csv')
    return load_and_clean_data(data_path)

try:
    df = load_data()
    
    # Visualization selection
    viz_option = st.selectbox(
        "Choose Visualization",
        ["Rating Distribution", "Location Analysis", "Cuisine Analysis", 
         "Price Analysis", "Online Order vs Rating", "Votes vs Rating"]
    )
    
    st.divider()
    
    if viz_option == "Rating Distribution":
        st.subheader("Restaurant Rating Distribution")
        
        fig, ax = plt.subplots(figsize=(9, 4),squeeze= True)
        sns.histplot(df['clean_rate'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of Restaurant Ratings', fontsize=3, fontweight='bold')
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Rating", f"{df['clean_rate'].mean():.2f}")
        with col2:
            st.metric("Median Rating", f"{df['clean_rate'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df['clean_rate'].std():.2f}")
    
    elif viz_option == "Location Analysis":
        st.subheader("📍 Location-based Analysis")
        
        top_n = st.slider("Number of top locations to display", 5, 30, 15)
        
        tab1, tab2, tab3 = st.tabs(["Restaurant Count", "Avg Rating", "Avg Cost"])
        
        with tab1:
            location_counts = df['location'].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(12, 6))
            location_counts.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f'Top {top_n} Locations by Restaurant Count', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Restaurants', fontsize=12)
            ax.set_ylabel('Location', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            top_locations = df['location'].value_counts().head(top_n).index
            location_ratings = df[df['location'].isin(top_locations)].groupby('location')['clean_rate'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            location_ratings.plot(kind='barh', ax=ax, color='lightgreen')
            ax.set_title(f'Average Rating by Location (Top {top_n})', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average Rating', fontsize=12)
            ax.set_ylabel('Location', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            location_cost = df[df['location'].isin(top_locations)].groupby('location')['approx_cost(for two people)'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            location_cost.plot(kind='barh', ax=ax, color='lightcoral')
            ax.set_title(f'Average Cost by Location (Top {top_n})', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average Cost (₹)', fontsize=12)
            ax.set_ylabel('Location', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif viz_option == "Cuisine Analysis":
        st.subheader("🍜 Cuisine Analysis")
        
        tab1, tab2 = st.tabs(["Distribution", "Rating by Cuisine"])
        
        with tab1:
            cuisine_counts = df['cuisines'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 10))
            cuisine_counts.plot.pie(autopct='%1.1f%%', startangle=140, ax=ax)
            ax.set_title('Cuisine Distribution', fontsize=16, fontweight='bold')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            cuisine_ratings = df.groupby('cuisines')['clean_rate'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            cuisine_ratings.plot(kind='barh', ax=ax, color='orange')
            ax.set_title('Average Rating by Cuisine Type', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average Rating', fontsize=12)
            ax.set_ylabel('Cuisine', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif viz_option == "Price Analysis":
        st.subheader("💰 Price Analysis")
        
        tab1, tab2 = st.tabs(["Distribution", "Price vs Rating"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['approx_cost(for two people)'], bins=30, kde=True, ax=ax)
            ax.set_title('Distribution of Approximate Cost for Two People', fontsize=16, fontweight='bold')
            ax.set_xlabel('Cost (₹)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            st.pyplot(fig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Cost", f"₹{df['approx_cost(for two people)'].mean():.0f}")
            with col2:
                st.metric("Median Cost", f"₹{df['approx_cost(for two people)'].median():.0f}")
            with col3:
                st.metric("Max Cost", f"₹{df['approx_cost(for two people)'].max():.0f}")
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='approx_cost(for two people)', y='clean_rate', data=df, alpha=0.5, ax=ax)
            ax.set_title('Price vs Rating', fontsize=16, fontweight='bold')
            ax.set_xlabel('Approximate Cost for Two (₹)', fontsize=12)
            ax.set_ylabel('Rating', fontsize=12)
            st.pyplot(fig)
    
    elif viz_option == "Online Order vs Rating":
        st.subheader("📱 Online Order Impact on Ratings")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='online_order', y='clean_rate', data=df, ax=ax)
        ax.set_title('Rating Distribution by Online Order Availability', fontsize=16, fontweight='bold')
        ax.set_xlabel('Online Order Available', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        st.pyplot(fig)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            avg_with_online = df[df['online_order'] == 'Yes']['clean_rate'].mean()
            st.metric("Avg Rating (With Online Order)", f"{avg_with_online:.2f}")
        with col2:
            avg_without_online = df[df['online_order'] == 'No']['clean_rate'].mean()
            st.metric("Avg Rating (Without Online Order)", f"{avg_without_online:.2f}")
    
    elif viz_option == "Votes vs Rating":
        st.subheader("🗳️ Votes vs Rating Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='votes', y='clean_rate', data=df, alpha=0.5, ax=ax)
        ax.set_title('Votes vs Rating', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Votes', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        st.pyplot(fig)
        
        # Correlation
        correlation = df['votes'].corr(df['clean_rate'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
