# Customer Churn Predictor

An interpretable machine learning demo that predicts the likelihood a subscription
customer will churn — and explains *why*, in plain language, using the trained
model's own coefficients.

**Live demo:** _(add link once deployed)_
**App code:** `/app` — a Vite + React client-side demo, no backend required

## Why this project

Most churn dashboards make a prediction and stop there. For a business team to
actually act on it — deciding who to call, who gets a retention offer, who to
leave alone — they need to know *why* the model thinks a customer is at risk.
This project is built around that gap: it's not just a classifier, it's a
classifier that argues its own case.

The same pattern generalizes beyond telecom to any subscription business —
SaaS, fintech, streaming — anywhere "will this customer renew?" is the question.

## Dataset

The IBM/Kaggle **Telco Customer Churn** dataset: 7,043 customers of a telecom
provider, each labeled as churned or retained, with account, billing, and
service features. This demo trains on a real sample of that data
(`churn_data.csv`).

## Approach

- **Model:** Logistic regression (`scikit-learn`), chosen deliberately over a
  black-box model (e.g. gradient boosting) because its coefficients map
  directly to feature contributions — the same interpretability trade-off made
  in the companion [Tumor Risk Triage Assistant](../tumor-risk-triage-assistant)
  project.
- **Features:** tenure, monthly charges, total charges to date, contract type,
  internet service type, payment method, paperless billing, senior citizen
  status.
- **Class balance:** trained with `class_weight="balanced"` — churners are the
  minority class (~25%), and missing one is far more costly than a false alarm.
- **Metric that matters:** recall on the churn class, not raw accuracy. A
  missed churner is a lost customer; a false positive is just one unnecessary
  retention offer. Current model: **80% recall on churn**, with 69% overall
  accuracy — a deliberate trade favoring catching at-risk customers over
  overall precision.

## What the demo does

Adjust a customer's profile (tenure, charges, contract, internet service,
payment method) and the model recalculates churn probability live. Click
"Explain this result" and it generates a plain-language explanation computed
directly from the model's own weights — no external API call, so the demo
works standalone with zero backend dependency.

## Files

- `train_model.py` — loads the data, cleans it, trains the model, prints
  evaluation metrics, and exports the trained weights to `model_export.json`.
- `model_export.json` — the trained model's coefficients, scaler parameters,
  and evaluation metrics, in a format the React demo reads directly.
- `app/` — the deployable Vite + React demo (`ChurnPredictor.jsx` is the
  component; the model weights are embedded directly in it, mirroring the
  Tumor Risk Triage Assistant's architecture).

## Disclaimer

Portfolio demo built on a public research dataset. Not a production system.
