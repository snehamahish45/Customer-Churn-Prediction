import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocessing import load_data, clean_data, prepare_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "static",
    "images"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

print("=" * 60)
print("Loading dataset...")

df = load_data(DATA_PATH)

df = clean_data(df)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

print("Dataset loaded successfully.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = preprocessor.get_feature_names_out()

print("Number of processed features:", len(feature_names))
# =========================================================
# CREATE MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "CatBoost": CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1, 3],
        random_seed=42,
        verbose=False
    )
}


# =========================================================
# TRAIN MODELS
# =========================================================

results = []

best_model = None
best_score = 0
best_name = ""


for name, model in models.items():

    print("=" * 60)
    print(f"Training {name}")

    # -----------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------

    # ==========================================
    # Cross Validation
    # ==========================================

    if name != "CatBoost":

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        print("Cross Validation Scores:", scores)
        print("Mean CV Accuracy:", scores.mean())

    else:

        print("Cross Validation skipped for CatBoost.")

    # ==========================================
    # Train Model
    # ==========================================

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc = roc_auc_score(
        y_test,
        y_prob
    )

    # -----------------------------------------------------
    # Save Results
    # -----------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc

    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc, 4))

    # -----------------------------------------------------
    # Select Best Model
    # -----------------------------------------------------

    if roc > best_score:

        best_score = roc

        best_model = model

        best_name = name


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# =========================================================
# SAVE MODEL RESULTS
# =========================================================

results_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "model_results.csv"
    ),
    index=False
)


# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )
)


# =========================================================
# SAVE PREPROCESSOR
# =========================================================

joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )
)


# =========================================================
# ROC CURVE
# =========================================================

plt.figure()

RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title(
    f"ROC Curve - {best_name}"
)

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "roc_curve.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("ROC curve saved successfully.")


# =========================================================
# CONFUSION MATRIX
# =========================================================

plt.figure()

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

plt.title(
    f"Confusion Matrix - {best_name}"
)

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "confusion_matrix.png"
    ),
    bbox_inches="tight"
)

plt.close()
print("Confusion matrix saved successfully.")

# =========================================================
# SHAP SUMMARY PLOT
# =========================================================

print("=" * 60)
print("Creating SHAP summary plot...")

try:

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    if hasattr(best_model, "coef_"):

        print("Using LinearExplainer for Logistic Regression...")

        explainer = shap.LinearExplainer(
            best_model,
            X_train
        )

        shap_values = explainer.shap_values(X_test)

    # -----------------------------------------------------
    # Tree Models
    # -----------------------------------------------------

    elif hasattr(best_model, "feature_importances_"):

        print("Using TreeExplainer for tree-based model...")

        explainer = shap.TreeExplainer(
            best_model
        )

        shap_values = explainer.shap_values(
            X_test
        )

        # Binary classification
        if isinstance(shap_values, list):

            shap_values = shap_values[1]

    else:

        raise Exception(
            "Unsupported model for SHAP."
        )


    # -----------------------------------------------------
    # SHAP Summary Plot
    # -----------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            IMAGE_DIR,
            "shap_summary.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP summary plot saved successfully."
    )

except Exception as e:

    print(
        "SHAP generation failed:",
        e
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("=" * 60)
print("Creating feature importance...")


try:
    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    if hasattr(best_model, "coef_"):

        importance = best_model.coef_[0]

        feature_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        feature_df["Absolute Importance"] = (
            feature_df["Importance"].abs()
        )

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    elif hasattr(best_model, "feature_importances_"):

        importance = best_model.feature_importances_

        feature_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        feature_df["Absolute Importance"] = (
            feature_df["Importance"].abs()
        )

    else:

        feature_df = None


    if feature_df is not None:

        feature_df = feature_df.sort_values(
            "Absolute Importance",
            ascending=False
        )

        # Save feature importance CSV

        feature_df.to_csv(
            os.path.join(
                MODEL_DIR,
                "feature_importance.csv"
            ),
            index=False
        )

        # Top 15 features

        top_features = feature_df.head(15)

        plt.figure(figsize=(10, 7))

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Absolute Importance"][::-1]
        )

        plt.xlabel("Importance")

        plt.ylabel("Feature")

        plt.title(
            f"Top Feature Importance - {best_name}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                IMAGE_DIR,
                "feature_importance.png"
            ),
            bbox_inches="tight"
        )

        plt.close()

        print("Feature importance saved successfully.")

except Exception as e:

    print(
        "Feature importance generation failed:",
        e
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n")
print("=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)

print("Best Model :", best_name)

print(
    "Best ROC-AUC:",
    round(best_score, 4)
)

print("Model saved successfully.")

print("Preprocessor saved successfully.")

print("Results saved successfully.")

print("Images saved successfully.")

print("=" * 60)