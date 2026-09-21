"""
Versatile AI/ML Portfolio Project: Customer Churn Prediction
--------------------------------------------------------------
Goal: Predict which customers are likely to cancel their subscription
(churn), using a real telecom customer dataset. Domain-agnostic pattern --
this same approach (interpretable classifier + LLM-free explanation layer)
applies to churn prediction at any subscription business: SaaS, telecom,
fintech, streaming, edtech.

Dataset: IBM/Kaggle "Telco Customer Churn" dataset (WA_Fn-UseC_-Telco-
Customer-Churn.csv), a widely-used real-world churn benchmark of 7,043
customers. This script uses a 488-row sample of that dataset.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load data
df = pd.read_csv("churn_data.csv")

# 2. Clean up: TotalCharges has a few blank strings for brand-new customers
#    (tenure=0), which pandas reads as NaN once coerced to numeric.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. Target: 1 = churned, 0 = stayed
y = (df["Churn"] == "Yes").astype(int)

# 4. Feature selection: a mix of numeric usage/billing signals and
#    categorical contract/service signals -- the classic churn drivers.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "Contract", "InternetService", "PaymentMethod",
    "PaperlessBilling", "SeniorCitizen",
]

X_numeric = df[numeric_features].copy()
X_categorical = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)
X = pd.concat([X_numeric, X_categorical], axis=1)
feature_names = list(X.columns)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale numeric features (logistic regression is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train logistic regression
#    class_weight="balanced" because churners are the minority class,
#    and in a retention context, missing a likely-churner (false negative)
#    is far more costly than flagging a loyal customer for outreach.
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Evaluate
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
precision_churn = precision_score(y_test, y_pred, pos_label=1)
recall_churn = recall_score(y_test, y_pred, pos_label=1)
f1_churn = f1_score(y_test, y_pred, pos_label=1)
cm = confusion_matrix(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision (churn): {precision_churn:.3f}")
print(f"Recall (churn):    {recall_churn:.3f}  <-- most important for retention targeting")
print(f"F1 (churn):        {f1_churn:.3f}")
print("\nConfusion matrix (rows=actual, cols=predicted) [stay, churn]:")
print(cm)
print("\n", classification_report(y_test, y_pred, target_names=["stayed", "churned"]))

# 9. Export everything needed to reproduce this model's predictions
#    in plain JavaScript for a client-side demo (same pattern as the
#    tumor risk triage project).
export = {
    "feature_names": feature_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "metrics": {
        "accuracy": acc,
        "precision_churn": precision_churn,
        "recall_churn": recall_churn,
        "f1_churn": f1_churn,
        "confusion_matrix": cm.tolist(),
        "test_set_size": int(len(y_test)),
        "training_set_size": int(len(y_train)),
    },
}

with open("model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nExported model to model_export.json")
print(f"\nFeature count: {len(feature_names)}")
print("Features:", feature_names)
