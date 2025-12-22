import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# FILE PATH
# -----------------------------
FILE_PATH = r"C:\Users\haris\Desktop\New folder (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

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
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# ENCODING CATEGORICAL FEATURES
# -----------------------------
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MODEL (GOOD BASELINE)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL (PICKLE)
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully at: {MODEL_PATH}")
