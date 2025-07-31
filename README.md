# ML-Project-pet-disease-prediction-
ML-based Pet Disease Prediction using XGBoost trained on symptom data of cats and dogs. Early detection for veterinary care.

# Pet Disease Prediction using XGBoost

## Problem Statement
Diseases in cats and dogs pose serious risks. Early detection is key to preventing worsening symptoms and reducing mortality.

## Proposed Solution
This project uses a machine learning model to predict pet diseases based on user-reported symptoms using XGBoost.

## Dataset
- Source: GitHub (Cats and Dogs Diseases)
- Records: ~14,494 entries after cleaning
- Features: 4 symptoms → 1 disease label

## Preprocessing Steps
- Deduplication
- Handling missing values
- One-hot encoding (MultiLabelBinarizer)
- Label encoding (LabelEncoder)
- Stratified train-test split

## Model
- Algorithm: XGBoost
- Tuning: RandomizedSearchCV with 30 iterations
- Metrics: Accuracy, Precision, Recall, F1-Score

## How to Run
1. Install dependencies:  
   `pip install pandas numpy scikit-learn xgboost matplotlib`
2. Run:  
   - `python predictor.py` or  
   - Open `disease_predictor.ipynb` in Jupyter


