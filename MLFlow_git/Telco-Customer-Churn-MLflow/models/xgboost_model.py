import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# -----------------------------
# FILE PATH
# -----------------------------
FILE_PATH = r"C:\Users\haris\Desktop\New folder (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_xgboost_model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(FILE_PATH)

# -----------------------------
# DATA CLEANING
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill numeric NaNs with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# HANDLE CLASS IMBALANCE (XGBOOST WAY)
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# XGBOOST MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL + ENCODERS
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "label_encoders": label_encoders,
            "features": X.columns.tolist()
        },
        f
    )

print(f"✅ XGBoost model saved successfully at: {MODEL_PATH}")
