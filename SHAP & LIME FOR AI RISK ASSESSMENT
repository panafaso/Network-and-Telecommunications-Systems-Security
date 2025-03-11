import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import demographic_parity_difference
import matplotlib.pyplot as plt

# Define AI Risk Categories
AI_RISK_CATEGORIES = {
    "Unacceptable Risk": "Banned AI (e.g., Social Scoring, Emotion Recognition at Work)",
    "High Risk": "Requires strict regulation (e.g., AI in hiring, credit scoring, biometric ID)",
    "Transparency Risk": "Needs clear disclosure (e.g., AI chatbots, deepfakes)",
    "Minimal Risk": "No regulation needed (e.g., spam filters, AI in video games)"
}

def classify_ai_risk(ai_use_case):
    """Classifies an AI system based on risk category."""
    if ai_use_case in ["Social Scoring", "Emotion Recognition at Work", "Biometric Surveillance"]:
        return "Unacceptable Risk"
    elif ai_use_case in ["Hiring Decisions", "Credit Scoring", "Law Enforcement AI"]:
        return "High Risk"
    elif ai_use_case in ["AI Chatbots", "Deepfakes", "Personalized Ads"]:
        return "Transparency Risk"
    else:
        return "Minimal Risk"

# Example AI system use cases
ai_systems = ["Social Scoring", "Hiring Decisions", "AI Chatbots", "Spam Filtering"]
for system in ai_systems:
    classify_ai_risk(system)

# Creating a synthetic dataset for GDPR Compliance Analysis
X = pd.DataFrame({
    'Age': [25, 45, 35, 50],
    'Income': [40000, 80000, 60000, 90000],
    'Sensitive_Data': [1, 0, 1, 0]  # 1 means contains GDPR-sensitive data
})
y = [0, 1, 1, 0]  # AI decision: Approved (1) or Denied (0)

# Train an AI model
model = RandomForestClassifier()
model.fit(X, y)

# SHAP Explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)

# LIME Explanation
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X.values, feature_names=X.columns, class_names=['Denied', 'Approved'], mode='classification'
)
exp = explainer_lime.explain_instance(X.iloc[2].values, model.predict_proba)
exp.show_in_notebook()

# GDPR Compliance Analysis - Check for fairness
demographic_parity_difference(y, model.predict(X))

# Visualizing AI Risk Categories
plt.figure(figsize=(8, 5))
plt.bar(AI_RISK_CATEGORIES.keys(), [1, 2, 3, 4], color=['red', 'orange', 'blue', 'green'])
plt.xlabel("AI Risk Category")
plt.ylabel("Severity Level")
plt.title("AI Risk Classification under EU AI Act")
plt.show()
