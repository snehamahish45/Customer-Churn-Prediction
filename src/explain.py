import shap
import joblib
import matplotlib.pyplot as plt

from preprocessing import load_data, clean_data, prepare_data

# Load dataset
df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = clean_data(df)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

# Load trained model
model = joblib.load("../models/best_model.pkl")

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# SHAP for Logistic Regression
explainer = shap.LinearExplainer(model, X_train)

shap_values = explainer.shap_values(X_test)

# Create summary plot
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "../static/images/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Saved!")