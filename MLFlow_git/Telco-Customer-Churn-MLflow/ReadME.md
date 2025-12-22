# Telco Customer Churn Prediction – End-to-End MLflow Project

This project is an end-to-end Machine Learning solution to predict customer churn
using the IBM Telco Customer Churn dataset. It demonstrates Exploratory Data Analysis (EDA),
multiple machine learning models, experiment tracking using MLflow, and REST APIs using Flask.

---

## Problem Statement
Customer churn is a major business challenge in the telecom industry.
The objective of this project is to analyze customer behavior, identify churn-driving factors,
and build predictive models to help reduce churn.

---

## Dataset
IBM Telco Customer Churn Dataset

- Total records: 7043
- Target variable: Churn (Yes / No)
- Feature categories:
  - Demographics
  - Account information
  - Subscribed services
  - Billing and payment details

---

## End-to-End Workflow
1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature encoding and preprocessing
4. Model training
5. Model evaluation
6. Experiment tracking with MLflow
7. API development for analytics

---

## Project Structure
Telco-Customer-Churn-MLflow/

- eda.py – Exploratory Data Analysis
- eda_summary_report.txt – Text-based EDA summary
- mlflow_models.py – MLflow experiment tracking for all models
- apis.py – Flask APIs for EDA insights

models/
- logistic_regression.py
- decision_tree.py
- random_forest.py
- xgboost_model.py
- lightgbm_model.py

- requirements.txt
- README.md
- .gitignore

---

## Exploratory Data Analysis (EDA)
EDA was performed to understand customer behavior and churn patterns.

Key findings:
- Overall churn rate is approximately 26.5%
- Customers with month-to-month contracts churn more
- Lower tenure customers have a higher churn probability
- Higher monthly charges correlate with churn

Detailed insights are available in eda_summary_report.txt.

---

## Machine Learning Models
The following models were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

Class imbalance handling techniques:
- class_weight='balanced'
- scale_pos_weight for boosting models

---

## MLflow Experiment Tracking
MLflow is used to track:
- Model parameters
- Evaluation metrics
- Model versions

Each experiment logs:
- Hyperparameters
- Accuracy, Precision, Recall, and F1 Score

Note:
MLflow artifacts such as mlruns/, mlflow.db, and model binaries are excluded from GitHub
as per best practices.

---

## Flask APIs
A Flask-based REST API exposes EDA insights in JSON format.

Sample endpoints:
- /api/churn-distribution
- /api/gender-churn
- /api/contract-churn
- /api/tenure-distribution
- /api/correlation-heatmap
- /api/imbalance-to-balance-undersample
- /api/imbalance-to-balance-oversample

These APIs can be used for dashboards or frontend integration.

---

## How to Run the Project

Install dependencies:
pip install -r requirements.txt

Run MLflow experiments:
python mlflow_models.py
mlflow ui

Access MLflow UI at:
http://localhost:5000

Run Flask API:
python apis.py

API server runs at:
http://localhost:5000

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- MLflow
- Flask
- Git & GitHub

---

## Key Learnings
- Handling class imbalance in real-world datasets
- Comparing linear and tree-based ML models
- Using MLflow for experiment tracking
- Structuring production-ready ML projects
- Building analytics APIs using Flask

---

## Author
Harish A N
Data Scientist

---

## Future Enhancements
- Hyperparameter tuning with MLflow
- Model deployment using Docker or cloud services
- Interactive dashboards for churn analysis
- CI/CD pipeline for ML workflows
