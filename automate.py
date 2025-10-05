import os
import logging
from datetime import datetime

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

# Project structure
project_structure = {
    'data': {
        'raw': [],
        'processed': []
    },
    'src': {
        '__init__.py': '',
        'data_cleaning.py': '',
        'data_processing.py': '',
        'visualization.py': '',
        'utils.py': '',
        'models': {
            '__init__.py': '',
            'train.py': '',
            'predict.py': '',
            'evaluate.py': ''
        }
    },
    'streamlit': {
        'app.py': '',
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
    'tests': {
        'test_models.py': ''
    }
}

def create_structure(base_path, structure, current_path=''):
    """Recursively create directory structure."""
    for name, content in structure.items():
        path = os.path.join(base_path, current_path, name)
        
        if isinstance(content, dict):
            # If content is a dict, it's a directory
            try:
                os.makedirs(path, exist_ok=True)
                logging.info(f"Created directory: {path}")
                create_structure(base_path, content, os.path.join(current_path, name))
            except Exception as e:
                logging.error(f"Error creating directory {path}: {str(e)}")
        else:
            # If content is not a dict, it's a file
            try:
                # Only create file if it doesn't exist
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        pass  # Create empty file
                    logging.info(f"Created file: {path}")
            except Exception as e:
                logging.error(f"Error creating file {path}: {str(e)}")

def create_requirements():
    """Create requirements.txt with initial dependencies."""
    requirements = [
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "streamlit",
        "scikit-learn",
        "pytest",
        "kaggle"
    ]
    
    try:
        with open('requirements.txt', 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        logging.info("Created requirements.txt")
    except Exception as e:
        logging.error(f"Error creating requirements.txt: {str(e)}")

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """
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

# VS Code
.vscode/

# Environment
.env
.venv
venv/
ENV/

# Logs
logs/
*.log

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
"""
    
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        logging.info("Created .gitignore")
    except Exception as e:
        logging.error(f"Error creating .gitignore: {str(e)}")

def main():
    """Main function to set up project structure."""
    logging.info("Starting project setup...")
    
    # Get the current working directory
    base_path = os.getcwd()
    logging.info(f"Setting up project in: {base_path}")
    
    # Create project structure
    create_structure(base_path, project_structure)
    
    # Create requirements.txt
    create_requirements()
    
    # Create .gitignore
    create_gitignore()
    
    # Create placeholder files to keep empty directories
    for dir_path in ['data/raw', 'data/processed', 'models', 'EDA_Graphs']:
        # Ensure full path is used and directory exists
        full_dir_path = os.path.join(base_path, dir_path)
        os.makedirs(full_dir_path, exist_ok=True)
        
        # Create .gitkeep file
        gitkeep_path = os.path.join(full_dir_path, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass
        logging.info(f"Created .gitkeep in {dir_path}")
    
    logging.info("Project setup completed successfully!")
    
    
    
if __name__ == "__main__":
    main()