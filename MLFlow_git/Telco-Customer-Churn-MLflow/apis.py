"""
Flask API for Telco Customer Churn EDA
Provides JSON responses for EDA, statistics, imbalance handling
"""

# =============================================================================
# IMPORTS
# =============================================================================
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# APP CONFIG
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# FILE PATH (YOUR DATASET)
# =============================================================================
FILE_PATH = r"C:\Users\haris\Desktop\New folder (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"

# =============================================================================
# LOAD DATA ONCE
# =============================================================================
print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

# Data cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print("Dataset loaded successfully!")

# =============================================================================
# ROOT ENDPOINT
# =============================================================================
@app.route("/")
def index():
    return jsonify({
        "message": "Telco Customer Churn EDA API",
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "endpoints": [
            "/api/churn-distribution",
            "/api/churn-count",
            "/api/gender-churn",
            "/api/senior-citizen-churn",
            "/api/partner-dependents",
            "/api/tenure-distribution",
            "/api/monthly-charges",
            "/api/total-charges",
            "/api/contract-churn",
            "/api/payment-method-churn",
            "/api/internet-service-churn",
            "/api/service-features",
            "/api/correlation-heatmap",
            "/api/tenure-vs-monthly-charges",
            "/api/churn-rates",
            "/api/statistical-summary",
            "/api/numeric-distributions",
            "/api/null-values-count",
            "/api/outliers-count",
            "/api/imbalance-to-balance-undersample",
            "/api/imbalance-to-balance-oversample"
        ]
    })

# =============================================================================
# BASIC EDA APIs
# =============================================================================
@app.route("/api/churn-distribution")
def churn_distribution():
    counts = df["Churn"].value_counts()
    return jsonify({
        "labels": counts.index.tolist(),
        "values": counts.values.tolist(),
        "percentages": [round(v / len(df) * 100, 2) for v in counts.values]
    })

@app.route("/api/churn-count")
def churn_count():
    counts = df["Churn"].value_counts()
    return jsonify({
        "labels": counts.index.tolist(),
        "values": counts.values.tolist()
    })

@app.route("/api/gender-churn")
def gender_churn():
    tab = pd.crosstab(df["gender"], df["Churn"])
    return jsonify(tab.to_dict())

@app.route("/api/senior-citizen-churn")
def senior_citizen_churn():
    tab = pd.crosstab(df["SeniorCitizen"], df["Churn"])
    return jsonify(tab.to_dict())

@app.route("/api/partner-dependents")
def partner_dependents():
    return jsonify({
        "partner": pd.crosstab(df["Partner"], df["Churn"]).to_dict(),
        "dependents": pd.crosstab(df["Dependents"], df["Churn"]).to_dict()
    })

# =============================================================================
# NUMERIC DISTRIBUTIONS
# =============================================================================
@app.route("/api/tenure-distribution")
def tenure_distribution():
    hist, bins = np.histogram(df["tenure"].dropna(), bins=40)
    return jsonify({
        "bins": bins.tolist(),
        "values": hist.tolist()
    })

@app.route("/api/monthly-charges")
def monthly_charges():
    hist, bins = np.histogram(df["MonthlyCharges"].dropna(), bins=40)
    return jsonify({
        "bins": bins.tolist(),
        "values": hist.tolist()
    })

@app.route("/api/total-charges")
def total_charges():
    clean = df["TotalCharges"].dropna()
    hist, bins = np.histogram(clean, bins=40)
    return jsonify({
        "bins": bins.tolist(),
        "values": hist.tolist()
    })

# =============================================================================
# CATEGORICAL ANALYSIS
# =============================================================================
@app.route("/api/contract-churn")
def contract_churn():
    return jsonify(pd.crosstab(df["Contract"], df["Churn"]).to_dict())

@app.route("/api/payment-method-churn")
def payment_method_churn():
    return jsonify(pd.crosstab(df["PaymentMethod"], df["Churn"]).to_dict())

@app.route("/api/internet-service-churn")
def internet_service_churn():
    return jsonify(pd.crosstab(df["InternetService"], df["Churn"]).to_dict())

@app.route("/api/service-features")
def service_features():
    features = [
        "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","PaperlessBilling"
    ]
    result = {}
    for f in features:
        result[f] = pd.crosstab(df[f], df["Churn"]).to_dict()
    return jsonify(result)

# =============================================================================
# CORRELATION
# =============================================================================
@app.route("/api/correlation-heatmap")
def correlation_heatmap():
    numeric_df = df.select_dtypes(include=[np.number])
    return jsonify(numeric_df.corr().round(3).to_dict())

# =============================================================================
# SCATTER
# =============================================================================
@app.route("/api/tenure-vs-monthly-charges")
def tenure_vs_monthly_charges():
    return jsonify({
        "no_churn": {
            "x": df[df["Churn"] == "No"]["tenure"].tolist(),
            "y": df[df["Churn"] == "No"]["MonthlyCharges"].tolist()
        },
        "churn": {
            "x": df[df["Churn"] == "Yes"]["tenure"].tolist(),
            "y": df[df["Churn"] == "Yes"]["MonthlyCharges"].tolist()
        }
    })

# =============================================================================
# CHURN RATES
# =============================================================================
@app.route("/api/churn-rates")
def churn_rates():
    return jsonify({
        "contract": (df.groupby("Contract")["Churn"]
                     .apply(lambda x: (x == "Yes").mean() * 100)).round(2).to_dict(),
        "payment_method": (df.groupby("PaymentMethod")["Churn"]
                           .apply(lambda x: (x == "Yes").mean() * 100)).round(2).to_dict()
    })

# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================
@app.route("/api/statistical-summary")
def statistical_summary():
    metrics = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary = {}
    for m in metrics:
        summary[m] = {
            "no_churn_mean": round(df[df["Churn"] == "No"][m].mean(), 2),
            "churn_mean": round(df[df["Churn"] == "Yes"][m].mean(), 2)
        }
    return jsonify(summary)

# =============================================================================
# NULL VALUES COUNT (NEW)
# =============================================================================
@app.route("/api/null-values-count")
def null_values_count():
    nulls = df.isnull().sum()
    return jsonify({
        "total_nulls": int(nulls.sum()),
        "by_column": nulls.to_dict()
    })

# =============================================================================
# OUTLIERS COUNT (NEW – IQR METHOD)
# =============================================================================
@app.route("/api/outliers-count")
def outliers_count():
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = {}

    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers[col] = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())

    return jsonify(outliers)

# =============================================================================
# UNDERSAMPLING (NEW)
# =============================================================================
@app.route("/api/imbalance-to-balance-undersample")
def undersample():
    counts = df["Churn"].value_counts()
    minority = counts.idxmin()
    majority = counts.idxmax()

    df_min = df[df["Churn"] == minority]
    df_maj = df[df["Churn"] == majority].sample(len(df_min), random_state=42)

    balanced = pd.concat([df_min, df_maj])

    return jsonify({
        "method": "undersampling",
        "before": counts.to_dict(),
        "after": balanced["Churn"].value_counts().to_dict()
    })

# =============================================================================
# OVERSAMPLING (NEW)
# =============================================================================
@app.route("/api/imbalance-to-balance-oversample")
def oversample():
    counts = df["Churn"].value_counts()
    minority = counts.idxmin()
    majority = counts.idxmax()

    df_min = df[df["Churn"] == minority].sample(len(df[df["Churn"] == majority]),
                                                replace=True,
                                                random_state=42)
    df_maj = df[df["Churn"] == majority]

    balanced = pd.concat([df_maj, df_min])

    return jsonify({
        "method": "oversampling",
        "before": counts.to_dict(),
        "after": balanced["Churn"].value_counts().to_dict()
    })

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

