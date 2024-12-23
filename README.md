# Credit Card Fraud Detection Analysis

## Overview

This project focuses on developing a machine learning model for detecting credit card fraud. As online transactions increase, so does the risk of fraudulent activities. This analysis leverages a comprehensive dataset to identify patterns in fraudulent behavior and implement effective detection mechanisms.

## Dataset

The dataset contains 1,852,394 entries with features such as user demographics, transaction details, and fraud indicators. The target variable is `is_fraud`, indicating whether a transaction is fraudulent (1) or legitimate (0).

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Matplotlib
- Seaborn

## Data Preprocessing

- **Data Cleaning**: Removed irrelevant columns and converted date-time formats.
- **Feature Engineering**: Developed features such as age, geographical distance, and job sector.
- **Handling Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

## Model Development

- Implemented various machine learning models, including:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - LightGBM
- Achieved an AUC of 0.968 and recall of 0.861 through hyperparameter optimization and cross-validation.

## Results

The final model demonstrated robust performance metrics:
- **AUC**: 0.968
- **Precision**: 0.07
- **Recall**: 0.861
- **F1 Score**: 0.13

## Business Insights

The analysis provided valuable insights for enhancing fraud prevention strategies, focusing on demographic and geographic risk factors to improve customer satisfaction and reduce false positives.
