import joblib
import matplotlib.pyplot as plt

from preprocessing import load_data, clean_data, prepare_data

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------

df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = clean_data(df)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

# -----------------------------
# Load Best Model
# -----------------------------

model = joblib.load("../models/best_model.pkl")

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("../models/confusion_matrix.png")

plt.show()

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve")

plt.savefig("../models/roc_curve.png")

plt.show()