# D100_Final_Project

## Background

In this project, I will focus on the violent crime rate in New York State. The main research question is: Which types of features (population, unemployment, urbanization, traffic count...) are most influential and important in determining violent crime rates in New York State? I will use Generalized Linear Model and Light Gradient Boosting Machine Model to conduct the violent crime rate prediction.

## Repository Structure

Final_Project/

- data/
  - data.py # Load and merge raw data
  - final_df.parquet # Final processed data
- eda_plot/
  - plotting.py # Plotting function for EDA
- model_analysis/
  - model_traning.py # THE MAIN FILE for Modelling and Evaluation
- src/
  - evaluation_prediction.py # Function for evaluating the GLM & LGBM model
  - feature_engineering.py # Log transformers
- test/
  - test_log.py # Unit tests for log transformers
- .gitignore # Ignore file configuration
- eda_cleaning.ipynb # Exploratory data analysis + Data cleaning
- .pre-commit-config.yaml # Pre-commit hooks
- environment.yml # Conda environment
- pyproject.toml # Project configuration
- README.md

## Installation

1. Open the file

- Method 1: Extract the zip file "d100*d400_code*{1728H}.zip"
- Method 2: Clone the repository "git clone <repo-url>"

2. Create conda environment

- conda env create -f environment.yml

3. Activate the environment

- conda activate final_project

4. Install project as package

- pip install .

5. Run the file: Model Training & Evaluation

- python -m model_analysis.model_training

## Data Preparation:

- Run the Jupyter notebook (`eda_cleaning.ipynb’) to perform EDA analysis and generate the parguet file

## Unit Test

- Run "pytest" in the terminal for log transformer test
