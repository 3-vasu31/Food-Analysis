# 🍕 Food Delivery Analysis Dashboard

A full-stack data science project that **cleans**, **explores**, and **models** the Zomato Bangalore restaurant dataset through an interactive Streamlit dashboard. Predict restaurant ratings or meal prices, explore visual insights, and understand exactly why each prediction is made all from your browser.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Feature Engineering — What and Why](#feature-engineering--what-and-why)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics — Interpretable by Design](#evaluation-metrics--interpretable-by-design)
- [Model Drawbacks & Limitations](#model-drawbacks--limitations)
- [Dashboard Pages — Demo](#dashboard-pages--demo)
  - [Home](#1-home)
  - [Overview](#2-overview)
  - [Data Analysis](#3-data-analysis)
  - [Insights](#4-insights)
  - [Predictions](#5-predictions)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Project Overview

The Indian food delivery market is massive and competitive. Restaurant owners need to understand what drives ratings and pricing, and customers want reliable recommendations. This project tackles both problems:

1. **Exploratory Data Analysis** — interactive charts that reveal patterns in locations, cuisines, pricing, and service features.
2. **Predictive Modeling** — three ML models that predict a restaurant's rating (1–5 stars) or approximate cost for two people (₹), with plain-English explanations of every prediction.

---

## Dataset

- **Source**: Zomato Bangalore Restaurants dataset (Kaggle)
- **Size**: ~51,000+ restaurant records
- **Key Columns**: restaurant name, location, cuisines, rating, votes, approximate cost for two, online order availability, table booking, restaurant type, and listing type.

---

## Feature Engineering — What and Why

Raw restaurant data is messy — inconsistent text, missing values, and high-cardinality categorical columns. Below is every cleaning and engineering step, and the reasoning behind it.

### 1. Dropping Irrelevant Columns

Columns `phone`, `dish_liked`, and `name` are dropped. Phone numbers are unique identifiers with no predictive power. `dish_liked` is a free-text field with too many nulls and variations. Restaurant `name` is a unique label, not a generalizable feature.

### 2. Location Cleaning & Matching

Locations in the raw data contain typos, embedded ratings (e.g., `"Koramangala (rated 4.1)"`), and inconsistencies between the `location` and `address` fields. The pipeline:

- Strips embedded ratings, extra whitespace, and invalid patterns from `location`.
- Extracts a candidate location from the `address` field (second-to-last comma-separated segment).
- Computes a **match score** (exact match → 3, substring → 2, word overlap → 1) between cleaned location and address-derived location, preferring the higher-confidence source.
- Keeps only the **top 75 locations** by frequency; everything else is bucketed as `"Rare Locations"`. This prevents the one-hot encoding from exploding into thousands of sparse columns while retaining the locations that have enough data to learn meaningful patterns.

**Why not just drop rare locations?** Bucketing them preserves data volume for training. For the modeling page, rare locations and unknowns are filtered out because their signal-to-noise ratio is too low.

### 3. Cuisine Grouping

The raw `cuisines` column contains 100+ unique values, often comma-separated lists like `"North Indian, Chinese, Biryani"`. Feeding these raw strings into a model would create extreme sparsity. Instead, each restaurant's cuisine list is mapped into **7 high-level groups** (Indian, Street Food, Desserts & Cafe, Asian, Middle Eastern, Western, Specialty) plus `Two-Or-Three-Cuisine`, `Multi-Cuisine`, and `Other`.

**Why group instead of one-hot encoding each cuisine?** With 100+ individual cuisines, one-hot encoding would create columns where most values are 0 — the model sees very few examples per cuisine and cannot learn reliable patterns. Grouping trades granularity for statistical power.

### 4. Restaurant Type Standardization

Restaurant types (`rest_type`) like `"Casual Dining"`, `"casual dining "`, `"Food Truck"`, and `"Kiosk"` are mapped to a controlled vocabulary of 9 categories (Quick Bites, Casual Dining, Cafe, Desserts, Bakery, Beverages, Fine Dining, Bar, Lounge, Food Court). Unrecognized types become `"Unknown"`.

**Why?** Consistent categories let the model learn that `"Food Truck"` and `"Kiosk"` behave similarly (both mapped to Quick Bites), rather than treating them as unrelated.

### 5. Rating Extraction

Ratings appear in multiple formats: `"4.1/5"`, `"Rated 3.8"`, `"NEW"`, `"-"`. A regex parser extracts the numeric value (1.0–5.0) and replaces unparseable entries with the column median.

### 6. Cost Cleaning

`approx_cost(for two people)` contains commas (`"1,200"`) and occasional non-numeric entries. These are stripped and cast to float; nulls are filled with the median.

### 7. Binary Encoding

`online_order` and `book_table` are converted to binary 0/1 flags (`online_order_yes`, `book_table_yes`). These are naturally binary features — encoding them as integers lets the model use them directly.

### 8. One-Hot Encoding (for Modeling)

The categorical features (location, cuisines, rest_type, listed_in(type)) are one-hot encoded with `drop_first=True` to avoid the dummy-variable trap. This is done **after** all the cleaning above, so the number of dummy columns is manageable (~100 instead of thousands).

---

## Machine Learning Models

Three models are available, each with different strengths. You pick one from a dropdown and train it on demand — no waiting for all three.

### 1. Random Forest Regressor

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 20 |
| `min_samples_leaf` | 2 |

**How it works**: Builds 200 decision trees on random subsets of the data and averages their predictions. Each tree is allowed to grow up to depth 20, and each leaf must have at least 2 samples (prevents overfitting to single data points).

**Strengths**: Handles non-linear relationships well; provides feature importance scores; robust to outliers; fast to train relative to neural networks.

**Drawbacks**: Can overfit if `max_depth` is too high; doesn't extrapolate well (predictions stay within the range of training data); the ensemble is a black box — you get feature importance but not a clear decision rule.

### 2. XGBoost Regressor

| Parameter | Value |
|---|---|
| `n_estimators` | 150 |
| `learning_rate` | 0.1 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

**How it works**: Gradient-boosted trees — each new tree corrects the errors of the previous ensemble. `subsample=0.8` and `colsample_bytree=0.8` introduce randomness (only 80% of rows and 80% of columns per tree) to reduce overfitting.

**Strengths**: Often the most accurate model on tabular data; handles missing values internally; built-in regularization; provides feature importance.

**Drawbacks**: More hyperparameters to tune than Random Forest; slower to train due to sequential boosting; requires the `xgboost` package (falls back to a tuned Random Forest if not installed); still a black-box model.

### 3. Neural Network (MLP Regressor)

| Parameter | Value |
|---|---|
| Architecture | 512 → 256 → 128 → 64 → 32 neurons |
| Activation | ReLU |
| Solver | Adam |
| Max iterations | 500 |
| Early stopping | Yes (10% validation split) |

**How it works**: A multi-layer perceptron with 5 hidden layers. Features are **standardized** (StandardScaler) before feeding into the network — this is critical because neural networks are sensitive to feature scale. Early stopping monitors a held-out 10% validation set and halts training when performance stops improving.

**Strengths**: Can learn highly non-linear and interaction-heavy patterns that tree models miss; flexible architecture.

**Drawbacks**: Slowest to train of the three; requires feature scaling (an extra preprocessing step); no built-in feature importance; prone to overfitting on small/medium datasets; results can vary with random initialization; harder to interpret.

---

## Evaluation Metrics — Interpretable by Design

Instead of technical metrics like RMSE or MAPE, this project uses three metrics chosen specifically because they answer plain questions:

| Metric | Question it Answers | How to Read It |
|---|---|---|
| **R² Score** | *"How much of the variation does the model explain?"* | 1.0 = perfect; 0.0 = no better than predicting the average every time. An R² of 0.75 means the model explains 75% of why ratings/prices differ. |
| **MAE (Mean Absolute Error)** | *"On average, how far off is each prediction?"* | For ratings: "off by 0.3 stars on average." For cost: "off by ₹150 on average." Lower is better. |
| **Accuracy %** | *"What percentage of predictions are close enough to be useful?"* | Measures how many predictions fall within ±0.5 stars (for ratings) or ±₹200 (for cost). An accuracy of 85% means 85 out of 100 predictions are within that tolerance. |

**Why not RMSE?** RMSE penalizes large errors more, but its units (squared, then rooted) are harder to explain to a non-technical audience. MAE is in the same unit as the target — "0.3 stars" or "₹150" — which is immediately meaningful.

**Why not MAPE?** MAPE breaks down when actual values are near zero and gives disproportionate weight to low-value predictions. Accuracy % within a tolerance is more intuitive: *"85% of the time, the model is within half a star."*

---

## Model Drawbacks & Limitations

1. **Data is Bangalore-specific** — the models learn Bangalore's restaurant landscape. They will not generalize to other cities without retraining.
2. **Temporal staleness** — the dataset is a snapshot in time. Restaurant ratings and prices change; the model doesn't account for trends.
3. **Cuisine grouping loses nuance** — a "North Indian" restaurant and a "Biryani" restaurant are both mapped to "Indian." The model can't distinguish sub-cuisine preferences.
4. **No text features** — reviews, menu descriptions, and dish names are not used. Adding NLP features could improve accuracy but would increase complexity.
5. **Location as a proxy** — location captures neighborhood effects (affluence, competition density) but doesn't explicitly model them. Two restaurants in the same location can have vastly different ratings.
6. **Regression on ordinal data** — ratings are ordinal (discrete: 1.0, 1.5, …, 5.0) but we treat them as continuous. This works in practice but the model can predict values like 3.73 that don't exist in the raw data.

---

## Dashboard Pages — Demo

### 1. Home

The landing page introduces the dashboard with a summary of what you can do: explore data, visualize trends, discover insights, and make predictions.

### 2. Overview

A bird's-eye view of the cleaned dataset:

- **Key metrics**: total restaurants, unique locations, cuisine types, average rating, average cost.
- **Column information**: data types, null counts.
- **Statistical summary**: mean, std, min/max for numeric columns.
- **Categorical distributions**: top locations, cuisines, online order and table booking availability as bar charts.
- **Sample data** viewer and a **CSV download** button for the cleaned dataset.

<!-- https://github.com/user-attachments/assets/demo-overview -->

<!-- https://github.com/user-attachments/assets/demo-overview -->
https://github.com/3-vasu31/Food-Analysis/blob/main/DemoVideos/OverviewPage.mp4

<!-- <video src="DemoVideos\OverviewPage.mp4" controls width="100%"></video> -->


<!-- <video width="600" controls>
  <source src="DemoVideos\OverviewPage.mp4" type="video/mp4">
</video> -->

![Overview Demo](DemoVideos\OverviewPage__01.gif)

### 3. Data Analysis

Interactive visualizations selectable via a dropdown:

- **Rating Distribution** — histogram with KDE overlay, plus mean/median/std metrics.
- **Location Analysis** — top-N locations by restaurant count, average rating, and average cost (tabbed).
- **Cuisine Analysis** — pie chart of cuisine distribution, bar chart of rating by cuisine.
- **Price Analysis** — cost distribution histogram, scatter plot of price vs rating.
- **Online Order vs Rating** — box plot comparing ratings for restaurants with and without online ordering.
- **Votes vs Rating** — scatter plot with correlation coefficient.

<video src="DemoVideos/DataAnalysis.mp4" controls width="100%"></video>

### 4. Insights

Deeper pattern discovery across five categories:

- **Restaurant Type Analysis** — distribution and rating comparison across types (Quick Bites, Casual Dining, Fine Dining, etc.).
- **Cuisine Patterns** — popularity vs quality vs price for each cuisine group.
- **Location Insights** — top-10 location comparison heatmap (rating, cost, votes).
- **Price-Rating Relationship** — ratings broken down by price category (Budget / Mid-Range / Premium / Luxury).
- **Service Features Impact** — how online ordering and table booking affect ratings and cost.

Each section includes auto-generated **Key Insights** callouts highlighting the most notable findings.

<video src="DemoVideos/Insights.mp4" controls width="100%"></video>

### 5. Predictions

The ML page with on-demand model training:

- **Model selector dropdown** — pick Random Forest, XGBoost, or Neural Network.
- **Train button** — trains only the selected model. Already-trained models are cached and load instantly on re-select.
- **Train All & Compare** — trains all three for a side-by-side comparison.
- **Status indicators** — green ✅ / amber ⏳ show which models are ready.
- **Metrics table** — R² Score, MAE, and Accuracy % for all trained models, with best values highlighted.
- **Comparison charts** — bar charts comparing Train vs Test for each metric (only shown when 2+ models are trained).
- **Plain-English interpretation** — expandable panels explaining what each model's numbers mean in words.
- **Actual vs Predicted plot** — scatter plot for the best model.
- **Feature Importance** — aggregated importance bar chart for tree-based models.
- **Make a Prediction** — fill in location, cuisine, restaurant type, online order, table booking, and cost/rating → get an instant prediction with a "What does this mean?" explanation.

<video src="DemoVideos/Predictiions01.mp4" controls width="100%"></video>

---

## Project Structure

```
Food-Analysis/
├── data/
│   └── zomato.csv/
│       └── zomato.csv            # Raw Zomato dataset
├── streamlit/
│   ├── app.py                    # Main Streamlit app (Home page)
│   ├── config.py                 # Configuration constants
│   ├── utils.py                  # Data cleaning & feature engineering
│   └── pages/
│       ├── 01_Overview.py        # Dataset overview
│       ├── 02_Data_Analysis.py   # Interactive visualizations
│       ├── 03_Insights.py        # Pattern discovery
│       └── 04_Predictions.py     # ML model training & predictions
├── DemoVideos/                   # Demo recordings for each page
├── EDA_Graphs/                   # Static EDA graph exports
├── notebooks/                    # Jupyter notebooks (exploration)
├── Test_Model/                   # Model testing notebooks
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md                     # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/3-vasu31/Food-Analysis.git
cd Food-Analysis
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For XGBoost support (optional but recommended for the best model):

```bash
pip install xgboost
```

### 4. Verify the Dataset

Make sure the data file exists at:

```
data/zomato.csv/zomato.csv
```

If you downloaded the dataset separately from Kaggle, place the `zomato.csv` file inside `data/zomato.csv/`.

### 5. Run the Dashboard

```bash
cd streamlit
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

### 6. Navigate the Dashboard

Use the sidebar to switch between pages:

1. **Home** — introduction
2. **Overview** — dataset summary and download
3. **Data Analysis** — interactive charts
4. **Insights** — pattern discovery
5. **Predictions** — train models, compare metrics, and make predictions

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Dashboard** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **Feature Engineering** | One-Hot Encoding, StandardScaler, regex-based text cleaning |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
