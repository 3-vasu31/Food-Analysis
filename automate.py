import os
import logging
import json
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Project structure matching STREAMLIT_README.md
project_structure = {
    'data': {
        'raw': [],
        'processed': [],
        'zomato.csv': []  # Directory for zomato.csv file
    },
    'streamlit': {
        'app.py': '',
        'utils.py': '',
        'config.py': '',
        'pages': {
            '01_Overview.py': '',
            '02_Data_Analysis.py': '',
            '03_Insights.py': '',
            '04_Predictions.py': ''
        }
    },
    'models': [],
    'EDA_Graphs': [],
    'notebooks': [],
    'logs': [],
    'Test_Model': {
        'test.ipynb': '',
        'test2.ipynb': '',
        'data': []
    }
}


def create_structure(base_path, structure, current_path=''):
    """
    Recursively create directory structure with enhanced logging.
    
    Args:
        base_path: Base directory path
        structure: Nested dictionary representing folder/file structure
        current_path: Current relative path being processed
    """
    for name, content in structure.items():
        path = os.path.join(base_path, current_path, name)
        
        if isinstance(content, dict):
            # Create directory and recurse
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"✓ Created directory: {path}")
                create_structure(base_path, content, os.path.join(current_path, name))
            except Exception as e:
                logger.error(f"✗ Error creating directory {path}: {str(e)}")
        elif isinstance(content, list):
            # Create directory (empty list = directory with no predefined files)
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"✓ Created directory: {path}")
            except Exception as e:
                logger.error(f"✗ Error creating directory {path}: {str(e)}")
        else:
            # Create file
            try:
                if not os.path.exists(path):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        if content:
                            f.write(content)
                    logger.info(f"✓ Created file: {path}")
                else:
                    logger.info(f"○ File already exists: {path}")
            except Exception as e:
                logger.error(f"✗ Error creating file {path}: {str(e)}")

def create_requirements():
    """Create requirements.txt with comprehensive dependencies for the project."""
    requirements = {
        # Core Data Processing
        "pandas>=2.0.0": "Data manipulation and analysis",
        "numpy>=1.24.0": "Numerical computing",
        
        # Visualization
        "matplotlib>=3.7.0": "Basic plotting library",
        "seaborn>=0.12.0": "Statistical data visualization",
        "plotly>=5.14.0": "Interactive visualizations",
        
        # Machine Learning
        "scikit-learn>=1.3.0": "Core ML algorithms and utilities",
        "xgboost>=2.0.0": "Gradient boosting framework",
        "lightgbm>=4.0.0": "Light gradient boosting machine",
        
        # Streamlit
        "streamlit>=1.28.0": "Web application framework",
        
        # Utilities
        "joblib>=1.3.0": "Model serialization",
        
        # Data Acquisition
        "kaggle>=1.5.16": "Kaggle dataset downloads",
        
        # Testing (Optional)
        "pytest>=7.4.0": "Testing framework",
    }
    
    try:
        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write("# Food Delivery Analysis - Project Dependencies\n")
            f.write(f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for package, description in requirements.items():
                f.write(f"{package}  # {description}\n")
        
        logger.info("✓ Created requirements.txt with all dependencies")
    except Exception as e:
        logger.error(f"✗ Error creating requirements.txt: {str(e)}")

def create_model_config():
    """Create model configuration file with different ML models and hyperparameters."""
    model_config = {
        "random_forest": {
            "display_name": "Random Forest",
            "class": "RandomForestRegressor",
            "configs": {
                "Default": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42
                },
                "Shallow Trees": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                },
                "Deep Trees": {
                    "n_estimators": 200,
                    "max_depth": 25,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42
                }
            }
        },
        "gradient_boosting": {
            "display_name": "Gradient Boosting",
            "class": "GradientBoostingRegressor",
            "configs": {
                "Default": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42
                },
                "Aggressive": {
                    "n_estimators": 150,
                    "learning_rate": 0.2,
                    "max_depth": 5,
                    "random_state": 42
                },
                "Conservative": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "random_state": 42
                }
            }
        },
        "xgboost": {
            "display_name": "XGBoost",
            "class": "XGBRegressor",
            "configs": {
                "Default": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                },
                "High Performance": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 8,
                    "random_state": 42
                }
            }
        },
        "lightgbm": {
            "display_name": "LightGBM",
            "class": "LGBMRegressor",
            "configs": {
                "Default": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "random_state": 42,
                    "verbose": -1
                },
                "Fast": {
                    "n_estimators": 50,
                    "learning_rate": 0.2,
                    "max_depth": -1,
                    "num_leaves": 15,
                    "random_state": 42,
                    "verbose": -1
                }
            }
        }
    }
    
    config_dir = 'streamlit'
    config_path = os.path.join(config_dir, 'model_config.json')
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=4)
        logger.info("✓ Created model_config.json with multiple model configurations")
    except Exception as e:
        logger.error(f"✗ Error creating model_config.json: {str(e)}")


def create_gitignore():
    """Create comprehensive .gitignore file."""
    gitignore_content = """# Food Analysis Project - Git Ignore Configuration

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.venv
venv/
ENV/

# Logs
logs/
*.log

# Data files (keep structure, ignore large data files)
data/raw/*
data/processed/*
data/zomato.csv/*.csv
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/zomato.csv/.gitkeep

# Model files
models/*.pkl
models/*.joblib

# OS specific
.DS_Store
Thumbs.db

# Streamlit cache
.streamlit/
"""
    
    try:
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write(gitignore_content.strip())
        logger.info("✓ Created .gitignore")
    except Exception as e:
        logger.error(f"✗ Error creating .gitignore: {str(e)}")

def create_gitkeep_files():
    """Create .gitkeep files to preserve empty directory structure in git."""
    directories = [
        'data/raw',
        'data/processed',
        'data/zomato.csv',
        'models',
        'EDA_Graphs',
        'notebooks',
        'Test_Model/data'
    ]
    
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            gitkeep_path = os.path.join(dir_path, '.gitkeep')
            with open(gitkeep_path, 'w', encoding='utf-8') as f:
                pass
            logger.info(f"✓ Created .gitkeep in {dir_path}")
        except Exception as e:
            logger.error(f"✗ Error creating .gitkeep in {dir_path}: {str(e)}")

def print_summary():
    """Print colorful setup summary."""
    summary = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🎉  FOOD ANALYSIS PROJECT SETUP COMPLETED SUCCESSFULLY! 🎉   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

📁 Project Structure:
   ✓ data/              - Dataset storage (raw, processed, zomato.csv)
   ✓ streamlit/         - Web application (app.py, utils.py, pages/)
   ✓ models/            - Saved ML models
   ✓ EDA_Graphs/        - Exploratory data analysis graphs
   ✓ notebooks/         - Jupyter notebooks
   ✓ Test_Model/        - Model testing notebooks
   ✓ logs/              - Application logs

📦 Configuration Files:
   ✓ requirements.txt   - Python dependencies
   ✓ .gitignore         - Git ignore rules
   ✓ model_config.json  - ML model hyperparameters

🚀 Next Steps:
   1. Install dependencies:
      pip install -r requirements.txt
   
   2. Place your dataset:
      data/zomato.csv/zomato.csv
   
   3. Run the Streamlit dashboard:
      cd streamlit
      streamlit run app.py
   
   4. Start developing! 💻

📋 Setup log saved to: {log_file}

═══════════════════════════════════════════════════════════════════
"""
    print(summary.format(log_file=log_file))

def main():
    """Main function to set up complete project structure."""
    logger.info("=" * 70)
    logger.info("FOOD ANALYSIS PROJECT - AUTOMATED SETUP")
    logger.info("=" * 70)
    
    # Get the current working directory
    base_path = os.getcwd()
    logger.info(f"Base Path: {base_path}")
    
    logger.info("\n🏗️  Creating project structure...")
    create_structure(base_path, project_structure)
    
    logger.info("\n📦 Creating requirements.txt...")
    create_requirements()
    
    logger.info("\n🚫 Creating .gitignore...")
    create_gitignore()
    
    logger.info("\n🤖 Creating model configuration...")
    create_model_config()
    
    logger.info("\n📌 Creating .gitkeep files...")
    create_gitkeep_files()
    
    logger.info("\n" + "=" * 70)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    # Print user-friendly summary
    print_summary()

if __name__ == "__main__":
    main()
