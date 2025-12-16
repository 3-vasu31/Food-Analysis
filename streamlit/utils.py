import pandas as pd
import numpy as np
import re



def load_and_clean_data(file_path, for_modeling=False):
    """
    Load and clean the Zomato dataset.
    
    Parameters:
    - file_path: Path to the CSV file
    - for_modeling: If True, removes rows with 'Unknown' values for better predictions
    
    Returns a cleaned dataframe ready for analysis and modeling.
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Make a copy
    df2 = df.copy()
    
    # Drop unnecessary columns
    df2.drop(columns=['phone', 'dish_liked', 'name'], inplace=True, errors='ignore')
    
    # Drop rows with missing location or address
    df2.dropna(subset=['location', 'address'], inplace=True)
    
    # Clean location and address
    df2 = clean_and_match_location(df2, Top_k_features=75)
    
    # Drop rows with null cuisines
    df2 = df2.drop(df2[df2['cuisines'].isnull()].index)
    
    # Clean cuisines
    df2['cuisines'] = df2['cuisines'].apply(map_cuisine)
    
    # Clean listed_in(type)
    top7_indices = df2["listed_in(type)"].value_counts()[:7]
    top7_indices_cleaned = [str(x).strip().lower() for x in top7_indices.index]
    df2['listed_in(type)'] = df2['listed_in(type)'].apply(
        clean_categorical_columns, common_values=top7_indices_cleaned
    )
    
    # Clean rest_type
    df2['rest_type'] = df2['rest_type'].apply(clean_rest_type)
    
    # Clean online_order
    df2 = clean_online_order(df2)
    
    # Clean rate
    df2 = clean_rate_column(df2)
    df2.drop('rate', axis=1, inplace=True, errors='ignore')
    
    # Clean book_table
    df2 = clean_book_table(df2)
    
    # Clean votes
    df2 = clean_votes(df2)
    
    # Clean approx_cost
    df2 = clean_approx_cost(df2)
    
    # Remove 'Unknown' values if this is for modeling (better accuracy)
    if for_modeling:
        original_count = len(df2)
        df2 = df2[
            (df2['rest_type'] != 'Unknown') & 
            (df2['listed_in(type)'] != 'Unknown') &
            (df2['location'] != 'Unknown') &
            (df2['location'] != 'Rare Locations')  # Also filter rare locations for better predictions
        ]
        removed_count = original_count - len(df2)
        print(f"Removed {removed_count} rows with 'Unknown' values for modeling ({removed_count/original_count*100:.1f}%)")
    
    return df2


def clean_and_match_location(df, Top_k_features):
    """
    Cleans and matches the 'location' and 'address' columns in the dataframe.
    """
    INVALID_PATTERNS = [
        r"\(''",
        r"\('",
        r"''",
        r"\(\s*''\s*\)",
    ]

    def contains_invalid_pattern(text):
        if text is None:
            return True
        text = str(text)
        for p in INVALID_PATTERNS:
            if re.search(p, text):
                return True
        return False

    def clean_location_text(loc):
        if pd.isna(loc) or contains_invalid_pattern(loc):
            return None
        loc = str(loc)
        loc = re.sub(r'\(\s*rated\s*[:\s]?\d+\.?\d*\s*\)', '', loc, flags=re.IGNORECASE)
        loc = re.sub(r'rated\s*\d+\.?\d*', '', loc, flags=re.IGNORECASE)
        loc = re.sub(r'\d+\.?\d*\s*/\s*5\b', '', loc, flags=re.IGNORECASE)
        loc = re.sub(r'(?i)rated[\s\S]*', '', loc).strip()
        loc = re.sub(r'\s+', ' ', loc).strip()
        return loc if len(loc) > 2 else None

    df['location_cleaned'] = df['location'].apply(clean_location_text)

    def clean_address(addr):
        if pd.isna(addr) or contains_invalid_pattern(addr):
            return None
        return str(addr).replace("\n", " ").strip()

    df['address_clean'] = df['address'].apply(clean_address)

    def extract_location_from_address(addr):
        if addr is None:
            return None
        parts = [p.strip() for p in addr.split(',')]
        noise_keywords = ['karnataka', 'bangalore', 'bengaluru', 'india', 'whitefield']
        if len(parts) >= 2:
            candidate = parts[-2].lower()
            if not contains_invalid_pattern(candidate) \
               and not re.search(r'\d{6}', candidate) \
               and not any(n in candidate for n in noise_keywords) \
               and len(candidate) > 3:
                return candidate
        return None

    df['address_location'] = df['address_clean'].apply(extract_location_from_address)

    def calculate_match_score(row):
        loc = row['location_cleaned']
        addr = row['address_location']
        if loc is None or addr is None:
            return 0
        if loc == addr:
            return 3
        if loc in addr:
            return 2
        if len(set(loc.split()) & set(addr.split())) > 0:
            return 1
        return 0

    df['location_match_score'] = df.apply(calculate_match_score, axis=1)

    def get_final_location(row):
        if row['location_cleaned'] is not None and row['location_match_score'] >= 2:
            return row['location_cleaned']
        if row['location_cleaned'] is not None:
            return row['location_cleaned']
        if row['address_location'] is not None:
            return row['address_location']
        return None

    df['location_final'] = df.apply(get_final_location, axis=1)
    df['location_final'] = df['location_final'].fillna('Unknown')

    top_n = Top_k_features
    location_counts = df['location_final'].value_counts()
    top_locations = location_counts.head(top_n).index

    df['location'] = df['location_final'].apply(
        lambda x: x if x in top_locations else 'Rare Locations'
    )

    df = df.drop(columns=[
        'location_cleaned',
        'address_clean',
        'address_location',
        'location_match_score',
        'location_final',
        'address'
    ])

    return df


def map_cuisine(cuisine):
    """Maps cuisine strings to cuisine groups."""
    if pd.isnull(cuisine):
        return "Other"

    cuisine_list = [c.strip().lower() for c in cuisine.split(",")]

    cuisine_groups = {
        'Indian': [
            'north indian', 'south indian', 'biryani', 'hyderabadi',
            'mughlai', 'punjabi', 'rajasthani', 'gujarati', 'bengali',
            'kerala', 'andhra', 'chettinad', 'mangalorean', 'goan',
            'indian', 'tandoor', 'curry', 'thali'
        ],
        'Street Food': [
            'street food', 'fast food', 'chaat', 'rolls', 'momos',
            'vada pav', 'pani puri', 'dosa', 'idli'
        ],
        'Desserts & Cafe': [
            'desserts', 'mithai', 'ice cream', 'bakery', 'beverages',
            'juice', 'shakes', 'tea', 'coffee', 'sweets', 'cafe'
        ],
        'Asian': [
            'chinese', 'thai', 'japanese', 'korean', 'vietnamese',
            'asian', 'pan asian', 'sushi', 'ramen', 'noodles',
            'dimsum', 'mongolian'
        ],
        'Middle Eastern': [
            'arabian', 'lebanese', 'mediterranean', 'turkish', 'persian',
            'kebab', 'shawarma', 'falafel', 'hummus'
        ],
        'Western': [
            'italian', 'mexican', 'american', 'continental', 'european',
            'pizza', 'burger', 'sandwich', 'pasta', 'steak', 'french'
        ],
        'Specialty': [
            'seafood', 'grill', 'bbq', 'barbecue', 'fusion',
            'healthy food', 'salad', 'vegan', 'vegetarian'
        ]
    }

    matched_groups = set()
    for group_name, keywords in cuisine_groups.items():
        for cuisine_item in cuisine_list:
            if cuisine_item in keywords:
                matched_groups.add(group_name)
                break

    if len(matched_groups) == 1:
        return list(matched_groups)[0]
    elif 2 <= len(matched_groups) <= 3:
        return 'Two-Or-Three-Cuisine'
    elif len(matched_groups) > 3:
        return 'Multi-Cuisine'
    else:
        return 'Other'


def clean_categorical_columns(value, common_values):
    """Standardizes categorical columns."""
    if pd.isnull(value):
        return 'Unknown'
    value_clean = str(value).strip().lower()
    if value_clean in common_values:
        return value_clean.title()
    else:
        return 'Unknown'


def clean_rest_type(rest_type):
    """Standardizes restaurant type categories."""
    if pd.isnull(rest_type):
        return 'Unknown'

    rest_type = rest_type.lower().strip()

    type_mapping = {
        'quick bites': 'Quick Bites',
        'casual dining': 'Casual Dining',
        'cafe': 'Cafe',
        'dessert parlor': 'Desserts',
        'dessert parlour': 'Desserts',
        'bakery': 'Bakery',
        'beverage shop': 'Beverages',
        'sweet shop': 'Desserts',
        'fine dining': 'Fine Dining',
        'bar': 'Bar',
        'pub': 'Bar',
        'lounge': 'Lounge',
        'food court': 'Food Court',
        'food truck': 'Quick Bites',
        'kiosk': 'Quick Bites'
    }

    for key, value in type_mapping.items():
        if key in rest_type:
            return value

    return 'Unknown'


def clean_online_order(df):
    """Normalizes online_order to Yes/No."""
    df["online_order"] = df["online_order"].astype(str).str.strip().str.lower()
    df["online_order"] = df["online_order"].map({"yes": "Yes", "no": "No"})
    mode_value = df["online_order"].mode()[0]
    df["online_order"] = df["online_order"].fillna(mode_value)
    return df


def clean_rate_column(df):
    """Extracts numeric ratings from various formats."""
    def parse_rate(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        match = re.search(r'(\d+\.?\d*)', x)
        if match:
            try:
                rating = float(match.group(1))
                if 1.0 <= rating <= 5.0:
                    return rating
            except:
                pass
        return np.nan

    df["clean_rate"] = df["rate"].apply(parse_rate)
    df["clean_rate"] = df.groupby("location")["clean_rate"].transform(
        lambda x: x.fillna(x.mean())
    )
    df["clean_rate"] = df["clean_rate"].round(1)
    return df


def clean_book_table(df):
    """Cleans book_table column."""
    df['book_table'] = df['book_table'].astype(str).str.strip().str.lower()
    df['book_table'] = df['book_table'].map({'yes': 'Yes', 'no': 'No'})
    df['book_table'] = df['book_table'].fillna('No')
    return df


def clean_votes(df):
    """Converts votes to numeric."""
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    df['votes'] = df['votes'].fillna(0).astype(int)
    return df


def clean_approx_cost(df):
    """Cleans approx_cost(for two people) column."""
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '')
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    df['approx_cost(for two people)'] = df.groupby('location')['approx_cost(for two people)'].transform(
        lambda x: x.fillna(x.median())
    )
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)
    return df
