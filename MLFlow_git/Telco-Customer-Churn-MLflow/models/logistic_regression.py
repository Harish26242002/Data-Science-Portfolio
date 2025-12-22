import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# FILE PATH
# -----------------------------
FILE_PATH = r"C:\Users\haris\Desktop\New folder (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_logistic_model.pkl")

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
    label_encoders[col] = le  # save encoders if needed later

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
# FEATURE SCALING (IMPORTANT FOR LOGISTIC)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# LOGISTIC REGRESSION MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
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
# SAVE MODEL + SCALER + ENCODERS
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoders": label_encoders,
            "features": X.columns.tolist()
        },
        f
    )

print(f"✅ Logistic Regression model saved successfully at: {MODEL_PATH}")
