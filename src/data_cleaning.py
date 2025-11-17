import pandas as pd
import numpy as np
import logging
from utils import clean_binary_columns, clean_categorical_columns
import re
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def drop_phone_column(df):
    """
    Drops the 'phone' column from the DataFrame if it exists.
    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    if 'phone' in df.columns:
        df.drop(['phone'], axis=1, inplace=True)
    return df


def clean_and_match_location(df, Top_k_features):

    '''
    Cleans and matches the 'location' and 'address' columns in the dataframe.
    Steps:
    1) Clean 'location' column by removing ratings and invalid patterns.
    2) Clean 'address' column by removing newlines and invalid patterns.
    3) Extract location from cleaned address.
    4) Calculate match score between cleaned location and extracted address location.
    5) Choose final location based on match score.
    6) Group rare locations into 'Other'.
    7) Drop intermediate columns used for processing.
    Parameters:
    df (pd.DataFrame): Input dataframe with 'location' and 'address' columns.
    Top_k_features (int): Number of top locations to retain; others grouped as 'Other'.
    Returns:
    pd.DataFrame: Dataframe with cleaned and matched 'location' column.
    '''
    INVALID_PATTERNS = [
        r"\(''",         # (''  
        r"\('",          # ('  
        r"''",           # ''  
        r"\(\s*''\s*\)", # ('') or similar
    ]

    def contains_invalid_pattern(text):
        if text is None:
            return True
        text = str(text)
        for p in INVALID_PATTERNS:
            if re.search(p, text):
                return True
        return False

    # 1) CLEAN LOCATION COLUMN
    def clean_location_text(loc):
        if pd.isna(loc) or contains_invalid_pattern(loc):
            return None

        loc = str(loc)

        # remove parenthetical ratings like (Rated 4.5)
        loc = re.sub(r'\(\s*rated\s*[:\s]?\d+\.?\d*\s*\)', '', loc, flags=re.IGNORECASE)

        # remove "rated 4.5"
        loc = re.sub(r'rated\s*\d+\.?\d*', '', loc, flags=re.IGNORECASE)

        # remove "4.5/5"
        loc = re.sub(r'\d+\.?\d*\s*/\s*5\b', '', loc, flags=re.IGNORECASE)

        # remove blocks starting with RATED
        loc = re.sub(r'(?i)rated[\s\S]*', '', loc).strip()

        # normalize whitespace
        loc = re.sub(r'\s+', ' ', loc).strip()

        return loc if len(loc) > 2 else None


    df['location_cleaned'] = df['location'].apply(clean_location_text)

    
    # 2) CLEAN ADDRESS COLUMN BEFORE EXTRACTION
    def clean_address(addr):
        if pd.isna(addr) or contains_invalid_pattern(addr):
            return None
        return str(addr).replace("\n", " ").strip()

    df['address_clean'] = df['address'].apply(clean_address)

    
    # 3) EXTRACT LOCATION FROM ADDRESS
    def extract_location_from_address(addr):
        if addr is None:
            return None

        parts = [p.strip() for p in addr.split(',')]

        noise_keywords = ['karnataka', 'bangalore', 'bengaluru', 'india', 'whitefield']

        # try second-to-last component
        if len(parts) >= 2:
            candidate = parts[-2].lower()
            if not contains_invalid_pattern(candidate) \
               and not re.search(r'\d{6}', candidate) \
               and not any(n in candidate for n in noise_keywords) \
               and len(candidate) > 3:
                return candidate

        return None

    df['address_location'] = df['address_clean'].apply(extract_location_from_address)

    
    # 4) MATCH SCORE
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

    
    # 5) CHOOSE FINAL LOCATION
    def get_final_location(row):
        if row['location_cleaned'] is not None and row['location_match_score'] >= 2:
            return row['location_cleaned']
        if row['location_cleaned'] is not None:
            return row['location_cleaned']
        if row['address_location'] is not None:
            return row['address_location']
        return None

    df['location_final'] = df.apply(get_final_location, axis=1)

    # replace None with 'Unknown'
    df['location_final'] = df['location_final'].fillna('Unknown')

    
    # 6) GROUP RARE LOCATIONS
    top_n = Top_k_features
    location_counts = df['location_final'].value_counts()
    top_locations = location_counts.head(top_n).index

    df['location'] = df['location_final'].apply(
        lambda x: x if x in top_locations else 'Unknown'
    )

 
    # 7) DROP INTERMEDIATE COLUMNS
 
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
    """
    Maps cuisine strings to cuisine groups.
    - If restaurant falls into 1 group: returns that group
    - If restaurant falls into 2-3 groups: returns 'Multi-Cuisine'
    - If falls into >3 groups or no match: returns 'Other'
    """
    if pd.isnull(cuisine):
        return "Other"
        
    # Clean & split
    cuisine_list = [c.strip().lower() for c in cuisine.split(",")]
    
    # Define cuisine groups
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
    
    # Find which groups this restaurant belongs to
    matched_groups = set()
    for group_name, keywords in cuisine_groups.items():
        for cuisine_item in cuisine_list:
            if cuisine_item in keywords:
                matched_groups.add(group_name)
                break
    
    # Determine output
    if len(matched_groups) == 1:
        # Single group
        return list(matched_groups)[0]
    elif 2 <= len(matched_groups) <= 3:
        # Multi-cuisine (2-3 groups)
        return 'Two-Or-Three-Cuisine'
    elif len(matched_groups) > 3:
        # Too many groups - likely noise
        return 'Multi-Cuisine'
    else:
        # No match found
        return 'Other'



def clean_online_order_and_book_table(df):
    
    # Clean 'online_orderd' and 'book_table' columns
    df['online_order'] = df['online_order'].apply(clean_binary_columns)
    df['book_table'] = df['book_table'].apply(clean_binary_columns)

    # Remove '/5' from 'rate' and convert to numeric
    df['rate'] = df['rate'].str.replace('/5', '')
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

    # Convert 'votes' to numeric
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

    # Remove commas from 'approx_cost(for two people)' and convert to numeric
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '')
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

    # # Remove listed_in column
    # df = df.drop(columns=['listed_in(type)'])
    
    return df


def clean_location(df):
        
    top_locations = df['location'].value_counts()[:20].index.tolist()

    df['location'] = df['location'].apply(clean_categorical_columns, common_values=top_locations)
    
    top7_indices = df["listed_in(type)"].value_counts()[:7].index
    df["listed_in(type)"] = df['listed_in(type)'].apply(clean_categorical_columns, common_values =top7_indices )
    
    return df


def clean_rate_column(df):
    def parse_rate(x):
        if pd.isna(x):
            return np.nan

        x = str(x).strip()

        # Case 1: "4.1/5"
        if "/" in x:
            try:
                return float(x.split("/")[0])
            except:
                return np.nan

        # Case 2: "Rated 4.0"
        if "Rated" in x:
            try:
                return float(x.replace("(Rated", "").strip())
            except:
                return np.nan

        # Case 3: anything else → NaN
        return np.nan

    # Apply parsing
    df["clean_rate"] = df["rate"].apply(parse_rate)

    # Fill NaN with average
    df["clean_rate"] = df.groupby("location")["clean_rate"].transform(
    lambda x: x.fillna(x.mean()))
    
    # Round to one decimal place
    df["clean_rate"] = df["clean_rate"].round(1)
   

    return df

def clean_online_order(df):
    """
    - Normalizes Yes/No.
    - Detects rating-like misplaced entries using regex and moves them to 'rate'.
    - Leaves gibberish as NaN (to be dropped or imputed later).
    """

    # Step 1: Normalize casing
    df["online_order"] = df["online_order"].astype(str).str.strip().str.lower()
    df["online_order"] = df["online_order"].map({"yes": "Yes", "no": "No"})

    # Step 2: Identify invalid entries
    mask_invalid = df["online_order"].isna()

    if "rate" in df.columns:
        for idx in df[mask_invalid].index:
            val = str(df.loc[idx, "online_order"])  # original bad value

            # Regex check: is it a rating? e.g. "4.1/5" or "Rated 4.0"
            if re.search(r"\d+(\.\d+)?", val):
                # move only rating-like values into 'rate'
                if pd.isna(df.loc[idx, "rate"]):
                    df.loc[idx, "rate"] = val
            else:
                # gibberish → leave as NaN for online_order
                df.loc[idx, "online_order"] = np.nan

    # Step 3: Fill NaN with mode (Yes/No)
    mode_value = df["online_order"].mode()[0]
    df["online_order"] = df["online_order"].fillna(mode_value)

    return df

