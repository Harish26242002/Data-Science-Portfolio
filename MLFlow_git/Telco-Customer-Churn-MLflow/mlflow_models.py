"""
MLflow Experiment Tracking
Telco Customer Churn – 5 Models Comparison

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
"""

# =============================================================================
# IMPORTS
# =============================================================================
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
import lightgbm as lgb

# =============================================================================
# CONFIG
# =============================================================================
FILE_PATH = r"C:\Users\haris\Desktop\New folder (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPERIMENT_NAME = "Telco_Churn_Models"

mlflow.set_experiment(EXPERIMENT_NAME)

# =============================================================================
# LOAD & PREPROCESS DATA
# =============================================================================
df = pd.read_csv(FILE_PATH)

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# METRIC FUNCTION
# =============================================================================
def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

# =============================================================================
# 1️⃣ LOGISTIC REGRESSION
# =============================================================================
with mlflow.start_run(run_name="Logistic_Regression"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mlflow.log_params({
        "model": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced"
    })
    mlflow.log_metrics(get_metrics(y_test, y_pred))
    mlflow.sklearn.log_model(model, "logistic_model")

# =============================================================================
# 2️⃣ DECISION TREE
# =============================================================================
with mlflow.start_run(run_name="Decision_Tree"):
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_params({
        "model": "DecisionTree",
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10
    })
    mlflow.log_metrics(get_metrics(y_test, y_pred))
    mlflow.sklearn.log_model(model, "decision_tree_model")

# =============================================================================
# 3️⃣ RANDOM FOREST
# =============================================================================
with mlflow.start_run(run_name="Random_Forest"):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })
    mlflow.log_metrics(get_metrics(y_test, y_pred))
    mlflow.sklearn.log_model(model, "random_forest_model")

# =============================================================================
# 4️⃣ XGBOOST
# =============================================================================
with mlflow.start_run(run_name="XGBoost"):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight
    })
    mlflow.log_metrics(get_metrics(y_test, y_pred))
    mlflow.xgboost.log_model(model, "xgboost_model")

# =============================================================================
# 5️⃣ LIGHTGBM
# =============================================================================
with mlflow.start_run(run_name="LightGBM"):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_params({
        "model": "LightGBM",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": scale_pos_weight
    })
    mlflow.log_metrics(get_metrics(y_test, y_pred))
    mlflow.lightgbm.log_model(model, "lightgbm_model")

print("✅ MLflow tracking completed for all 5 models")
