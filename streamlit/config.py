import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'zomato.csv', 'zomato.csv')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 15

# Feature engineering
TOP_K_LOCATIONS = 75

# Cuisine groups
CUISINE_GROUPS = {
    'Indian': ['North Indian', 'South Indian', 'Biryani', 'Andhra', 'Hyderabadi', 
               'Mughlai', 'Rajasthani', 'Bengali', 'Kerala', 'Lucknowi', 
               'Chettinad', 'Goan', 'Kashmiri', 'Awadhi', 'Punjabi'],
    'Street Food': ['Street Food', 'Fast Food', 'Burger', 'Pizza', 'Sandwich', 
                    'Rolls', 'Momos', 'Chaat', 'Kebab'],
    'Desserts & Cafe': ['Desserts', 'Bakery', 'Cafe', 'Coffee', 'Tea', 
                        'Ice Cream', 'Beverages', 'Juices', 'Mithai'],
    'Asian': ['Chinese', 'Thai', 'Japanese', 'Asian', 'Korean', 'Malaysian', 
              'Vietnamese', 'Burmese', 'Sushi', 'Ramen', 'Dim Sum'],
    'Middle Eastern': ['Arabian', 'Lebanese', 'Mediterranean', 'Turkish', 
                       'Middle Eastern', 'Iranian', 'Afghan'],
    'Western': ['Continental', 'Italian', 'American', 'Mexican', 'European', 
                'French', 'Spanish', 'Greek'],
    'Specialty': ['Seafood', 'BBQ', 'Steak', 'Salad', 'Healthy Food', 
                  'Organic', 'Vegan', 'Vegetarian']
}

# Streamlit page config
PAGE_TITLE = "🍽️ Zomato Restaurant Analysis Dashboard"
PAGE_ICON = "🍽️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Visualization colors
PRIMARY_COLOR = '#ff6b6b'
SECONDARY_COLOR = '#4ecdc4'
ACCENT_COLOR = '#45b7d1'
