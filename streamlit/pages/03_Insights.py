import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_and_clean_data

st.set_page_config(page_title="Insights", page_icon="💡", layout="wide")

st.title("💡 Key Insights")
st.write("Discover patterns in delivery zones, cuisines, and restaurant types")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'zomato.csv', 'zomato.csv')
    return load_and_clean_data(data_path)

try:
    df = load_data()
    
    # Insight Categories
    insight_choice = st.selectbox(
        "Select Insight Category",
        ["Restaurant Type Analysis", "Cuisine Patterns", "Location Insights", 
         "Price-Rating Relationship", "Service Features Impact"]
    )
    
    st.divider()
    
    if insight_choice == "Restaurant Type Analysis":
        st.subheader("🏪 Restaurant Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Restaurant Type Distribution**")
            rest_type_counts = df['rest_type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            rest_type_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Restaurant Types', fontweight='bold')
            ax.set_xlabel('Type')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("**Average Rating by Restaurant Type**")
            rest_type_rating = df.groupby('rest_type')['clean_rate'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            rest_type_rating.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Avg Rating by Type', fontweight='bold')
            ax.set_xlabel('Type')
            ax.set_ylabel('Average Rating')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.write("**💡 Key Insights:**")
        highest_rated_type = rest_type_rating.idxmax()
        st.info(f"• **{highest_rated_type}** restaurants have the highest average rating ({rest_type_rating.max():.2f})")
        most_common_type = rest_type_counts.idxmax()
        st.info(f"• **{most_common_type}** is the most common restaurant type with {rest_type_counts.max():,} restaurants")
    
    elif insight_choice == "Cuisine Patterns":
        st.subheader("🍜 Cuisine Patterns & Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cuisine Popularity**")
            cuisine_counts = df['cuisines'].value_counts()
            st.dataframe(cuisine_counts.head(10), use_container_width=True)
        
        with col2:
            st.write("**Cuisine Rating Analysis**")
            cuisine_rating = df.groupby('cuisines')['clean_rate'].mean().sort_values(ascending=False)
            st.dataframe(cuisine_rating.head(10), use_container_width=True)
        
        # Cuisine vs Price
        st.write("**Average Cost by Cuisine**")
        cuisine_cost = df.groupby('cuisines')['approx_cost(for two people)'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        cuisine_cost.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Top 10 Most Expensive Cuisines', fontweight='bold')
        ax.set_xlabel('Average Cost (₹)')
        ax.set_ylabel('Cuisine')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**💡 Key Insights:**")
        most_expensive_cuisine = cuisine_cost.idxmax()
        st.info(f"• **{most_expensive_cuisine}** is the most expensive cuisine type (₹{cuisine_cost.max():.0f} avg)")
        most_popular_cuisine = cuisine_counts.idxmax()
        st.info(f"• **{most_popular_cuisine}** is the most popular cuisine with {cuisine_counts.max():,} restaurants")
    
    elif insight_choice == "Location Insights":
        st.subheader("📍 Location-based Insights")
        
        # Top locations
        top_10_locations = df['location'].value_counts().head(10).index
        df_top_locations = df[df['location'].isin(top_10_locations)]
        
        # Location comparison
        location_stats = df_top_locations.groupby('location').agg({
            'clean_rate': 'mean',
            'approx_cost(for two people)': 'mean',
            'votes': 'mean'
        }).round(2)
        location_stats.columns = ['Avg Rating', 'Avg Cost', 'Avg Votes']
        location_stats = location_stats.sort_values('Avg Rating', ascending=False)
        
        st.write("**Top 10 Locations Comparison**")
        st.dataframe(location_stats, use_container_width=True)
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(location_stats, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Location Metrics Heatmap', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**💡 Key Insights:**")
        best_rated_location = location_stats['Avg Rating'].idxmax()
        st.info(f"• **{best_rated_location}** has the highest average rating ({location_stats.loc[best_rated_location, 'Avg Rating']:.2f})")
        most_expensive_location = location_stats['Avg Cost'].idxmax()
        st.info(f"• **{most_expensive_location}** is the most expensive area (₹{location_stats.loc[most_expensive_location, 'Avg Cost']:.0f} avg)")
    
    elif insight_choice == "Price-Rating Relationship":
        st.subheader("💰 Price vs Rating Analysis")
        
        # Create price bins
        df['price_category'] = pd.cut(df['approx_cost(for two people)'], 
                                      bins=[0, 300, 600, 1000, float('inf')],
                                      labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_rating = df.groupby('price_category')['clean_rate'].mean()
            fig, ax = plt.subplots(figsize=(8, 6))
            price_rating.plot(kind='bar', ax=ax, color='purple')
            ax.set_title('Average Rating by Price Category', fontweight='bold')
            ax.set_xlabel('Price Category')
            ax.set_ylabel('Average Rating')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            price_counts = df['price_category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            price_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title('Restaurant Distribution by Price', fontweight='bold')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation
        correlation = df['approx_cost(for two people)'].corr(df['clean_rate'])
        st.metric("Price-Rating Correlation", f"{correlation:.3f}")
        
        st.write("**💡 Key Insights:**")
        if correlation > 0.3:
            st.info("• Higher priced restaurants tend to have higher ratings (moderate positive correlation)")
        elif correlation < -0.3:
            st.info("• Higher priced restaurants tend to have lower ratings (moderate negative correlation)")
        else:
            st.info("• Price has weak correlation with ratings - other factors are more important")
    
    elif insight_choice == "Service Features Impact":
        st.subheader("🔧 Service Features Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Online Order Impact**")
            online_stats = df.groupby('online_order')['clean_rate'].agg(['mean', 'count'])
            fig, ax = plt.subplots(figsize=(8, 6))
            online_stats['mean'].plot(kind='bar', ax=ax, color='teal')
            ax.set_title('Avg Rating: Online Order', fontweight='bold')
            ax.set_xlabel('Online Order Available')
            ax.set_ylabel('Average Rating')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("**Table Booking Impact**")
            booking_stats = df.groupby('book_table')['clean_rate'].agg(['mean', 'count'])
            fig, ax = plt.subplots(figsize=(8, 6))
            booking_stats['mean'].plot(kind='bar', ax=ax, color='salmon')
            ax.set_title('Avg Rating: Table Booking', fontweight='bold')
            ax.set_xlabel('Table Booking Available')
            ax.set_ylabel('Average Rating')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Combined analysis
        st.write("**Combined Service Features**")
        combined_stats = df.groupby(['online_order', 'book_table'])['clean_rate'].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 6))
        combined_stats.plot(kind='bar', ax=ax)
        ax.set_title('Rating by Service Features Combination', fontweight='bold')
        ax.set_xlabel('Online Order')
        ax.set_ylabel('Average Rating')
        ax.legend(title='Table Booking')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**💡 Key Insights:**")
        online_yes_avg = df[df['online_order'] == 'Yes']['clean_rate'].mean()
        online_no_avg = df[df['online_order'] == 'No']['clean_rate'].mean()
        if online_yes_avg > online_no_avg:
            st.info(f"• Restaurants with online ordering have higher average ratings ({online_yes_avg:.2f} vs {online_no_avg:.2f})")
        else:
            st.info(f"• Restaurants without online ordering have slightly higher ratings ({online_no_avg:.2f} vs {online_yes_avg:.2f})")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
