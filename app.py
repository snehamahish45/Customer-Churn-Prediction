from flask import Flask, render_template, request, send_file
import os
import time
from datetime import datetime

import joblib
import pandas as pd


# =========================================================
# APP
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_results.csv"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "history.csv"
)

SHAP_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "static",
    "images",
    "shap_summary.png"
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print("==========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("==========================================")
    print("Model:", type(model).__name__)
    print(
        "Preprocessor:",
        type(preprocessor).__name__
    )

    if hasattr(model, "classes_"):
        print(
            "Model classes:",
            model.classes_
        )

    if hasattr(
        preprocessor,
        "feature_names_in_"
    ):
        print(
            "Preprocessor expects:",
            list(
                preprocessor.feature_names_in_
            )
        )

except Exception as e:

    raise RuntimeError(
        "Could not load the model/preprocessor. "
        "Check models/best_model.pkl and "
        "models/preprocessor.pkl."
    ) from e


# =========================================================
# DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    if "SeniorCitizen" in dataset.columns:

        dataset["SeniorCitizen"] = pd.to_numeric(
            dataset["SeniorCitizen"],
            errors="coerce"
        )

    if "tenure" in dataset.columns:

        dataset["tenure"] = pd.to_numeric(
            dataset["tenure"],
            errors="coerce"
        )

    if "MonthlyCharges" in dataset.columns:

        dataset["MonthlyCharges"] = pd.to_numeric(
            dataset["MonthlyCharges"],
            errors="coerce"
        )

    if "TotalCharges" in dataset.columns:

        dataset["TotalCharges"] = pd.to_numeric(
            dataset["TotalCharges"],
            errors="coerce"
        )

    return dataset

# =========================================================
# HISTORY
# =========================================================

def get_history():

    os.makedirs(
        os.path.dirname(HISTORY_PATH),
        exist_ok=True
    )

    required_columns = [
        "ID",
        "Timestamp",
        "Prediction",
        "Probability",
        "Confidence",
        "Risk",
    ]

    if not os.path.exists(
        HISTORY_PATH
    ):

        history = pd.DataFrame(
            columns=required_columns
        )

        history.to_csv(
            HISTORY_PATH,
            index=False
        )

        return history

    try:

        history = pd.read_csv(
            HISTORY_PATH
        )

    except Exception as e:

        print(
            "History loading error:",
            repr(e)
        )

        history = pd.DataFrame(
            columns=required_columns
        )

    for column in required_columns:

        if column not in history.columns:

            history[column] = ""

    return history[
        required_columns
    ]


# =========================================================
# MODEL METRICS
# =========================================================

def get_metrics():

    default_metrics = {

        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "roc": 0,

        "best_model":
            "Not Available",

        "model_count": 0,
    }

    if not os.path.exists(
        METRICS_PATH
    ):

        return default_metrics

    try:

        metrics = pd.read_csv(
            METRICS_PATH
        )

        if metrics.empty:

            return default_metrics

        if "ROC-AUC" in metrics.columns:

            best = metrics.loc[
                metrics["ROC-AUC"].idxmax()
            ]

        else:

            best = metrics.iloc[0]

        return {

            "accuracy":
                round(
                    float(
                        best["Accuracy"]
                    ) * 100,
                    2
                ),

            "precision":
                round(
                    float(
                        best["Precision"]
                    ) * 100,
                    2
                ),

            "recall":
                round(
                    float(
                        best["Recall"]
                    ) * 100,
                    2
                ),

            "f1":
                round(
                    float(
                        best["F1 Score"]
                    ) * 100,
                    2
                ),

            "roc":
                round(
                    float(
                        best["ROC-AUC"]
                    ) * 100,
                    2
                ),

            "best_model":
                str(best["Model"]),

            "model_count":
                len(metrics),
        }

    except Exception as e:

        print(
            "Metric loading error:",
            repr(e)
        )

        return default_metrics


# =========================================================
# TOP FEATURES
# =========================================================

def get_top_features():

    feature_importance_path = os.path.join(
        BASE_DIR,
        "models",
        "feature_importance.csv"
    )

    default_features = [
        "Tenure",
        "Contract",
        "Total Charges",
        "Internet Service",
    ]

    if not os.path.exists(
        feature_importance_path
    ):

        return default_features

    try:

        feature_df = pd.read_csv(
            feature_importance_path
        )

        if (
            "Feature"
            not in feature_df.columns
            or feature_df.empty
        ):

            return default_features

        raw_features = (
            feature_df
            .head(10)["Feature"]
            .astype(str)
            .tolist()
        )

        top_features = []

        for feature in raw_features:

            if feature == "num__tenure":

                display_name = "Tenure"

            elif feature.startswith(
                "cat__Contract_"
            ):

                display_name = "Contract"

            elif feature == "num__TotalCharges":

                display_name = "Total Charges"

            elif feature.startswith(
                "cat__InternetService_"
            ):

                display_name = "Internet Service"

            elif feature == "num__MonthlyCharges":

                display_name = "Monthly Charges"

            elif feature.startswith("cat__"):

                display_name = feature.replace(
                    "cat__",
                    ""
                )

                display_name = (
                    display_name
                    .replace("_", " ")
                )

            elif feature.startswith("num__"):

                display_name = feature.replace(
                    "num__",
                    ""
                )

                display_name = (
                    display_name
                    .replace("_", " ")
                )

            else:

                display_name = (
                    feature
                    .replace("_", " ")
                )

            if display_name not in top_features:

                top_features.append(
                    display_name
                )

            if len(top_features) >= 4:

                break

        if top_features:

            return top_features

    except Exception as e:

        print(
            "Feature importance error:",
            repr(e)
        )

    return default_features


# =========================================================
# PAGE DATA
# =========================================================

def get_page_data():

    metric_data = get_metrics()

    dataset = load_dataset()

    total = len(dataset)

    if "Churn" in dataset.columns:

        stay = int(
            (
                dataset["Churn"] == "No"
            ).sum()
        )

        churn = int(
            (
                dataset["Churn"] == "Yes"
            ).sum()
        )

    else:

        stay = 0
        churn = 0

    stay_rate = (
        round(
            stay / total * 100,
            1
        )
        if total
        else 0
    )

    churn_rate = (
        round(
            churn / total * 100,
            1
        )
        if total
        else 0
    )

    # -----------------------------------------------------
    # Revenue
    # -----------------------------------------------------

    if (
        "MonthlyCharges" in dataset.columns
        and "Churn" in dataset.columns
    ):

        churned_customers = dataset[
            dataset["Churn"] == "Yes"
        ]

        revenue_at_risk_value = float(
            churned_customers[
                "MonthlyCharges"
            ].sum()
        )

        total_monthly_revenue_value = float(
            dataset[
                "MonthlyCharges"
            ].sum()
        )

    else:

        revenue_at_risk_value = 0.0
        total_monthly_revenue_value = 0.0

    monthly_charges_at_risk = (
        f"${revenue_at_risk_value / 1000:.2f}K"
    )

    total_monthly_charges = (
        f"${total_monthly_revenue_value / 1000:.2f}K"
    )

    history = (
        get_history()
        .tail(10)
        .iloc[::-1]
    )

    top_features = get_top_features()

    return {

        "accuracy":
            metric_data["accuracy"],

        "precision":
            metric_data["precision"],

        "recall":
            metric_data["recall"],

        "f1":
            metric_data["f1"],

        "roc":
            metric_data["roc"],

        "best_model":
            metric_data["best_model"],

        "model_count":
            metric_data["model_count"],

        "input_features": 19,

        "total":
            total,

        "stay":
            stay,

        "churn":
            churn,

        "stay_rate":
            stay_rate,

        "churn_rate":
            churn_rate,

        "high_risk":
            churn,

        "avg_churn_probability":
            churn_rate,

        "dataset_churn_rate":
            churn_rate,

        "revenue_at_risk":
            monthly_charges_at_risk,

        "total_monthly_revenue":
            total_monthly_charges,

        "total_monthly_charges":
            total_monthly_charges,

        "prediction_time":
            "—",

        "top_risk_factor_count":
            len(top_features),

        "retention_opportunities":
            churn,

        "top_features":
            top_features,

        "history":
            history.to_dict(
                "records"
            ),
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        **get_page_data()
    )


# =========================================================
# RISK
# =========================================================

def get_risk(churn_percent):

    if churn_percent < 20:

        return (
            "Very Low Risk",
            "success"
        )

    if churn_percent < 40:

        return (
            "Low Risk",
            "warning"
        )

    if churn_percent < 60:

        return (
            "Medium Risk",
            "warning"
        )

    if churn_percent < 80:

        return (
            "High Risk",
            "danger"
        )

    return (
        "Critical Risk",
        "dark"
    )


# =========================================================
# CONFIDENCE
# =========================================================

def get_confidence_level(
    confidence
):

    if confidence >= 95:

        return "Very High"

    if confidence >= 85:

        return "High"

    if confidence >= 70:

        return "Medium"

    return "Low"


# =========================================================
# RECOMMENDATIONS
# =========================================================

def build_recommendations(
    customer,
    prediction,
    tenure,
    monthly_charges
):

    recommendations = []

    if prediction == 0:

        recommendations.append(
            f"Customer has remained with "
            f"the company for {tenure} months."
        )

        recommendations.append(
            "Continue loyalty rewards."
        )

        if (
            customer["Contract"]
            == "Month-to-month"
        ):

            recommendations.append(
                "Offer a discounted annual contract."
            )

        if monthly_charges < 70:

            recommendations.append(
                "Consider relevant premium "
                "add-on services."
            )

    else:

        recommendations.append(
            "Customer shows a high "
            "likelihood of churn."
        )

        if (
            customer["Contract"]
            == "Month-to-month"
        ):

            recommendations.append(
                "Offer an annual or two-year "
                "contract incentive."
            )

        if (
            customer["PaymentMethod"]
            == "Electronic check"
        ):

            recommendations.append(
                "Encourage automatic payment methods."
            )

        if monthly_charges > 80:

            recommendations.append(
                "Consider a personalized "
                "billing discount."
            )

        if (
            customer["InternetService"]
            == "Fiber optic"
        ):

            recommendations.append(
                "Review network quality "
                "and service experience."
            )

        if tenure < 12:

            recommendations.append(
                "Consider assigning a "
                "retention specialist."
            )

    return recommendations


# =========================================================
# FACTORS
# =========================================================

def build_factors(
    customer,
    tenure,
    monthly_charges,
    stay_percent
):

    positive_factors = []
    risk_factors = []

    if tenure >= 24:

        positive_factors.append(
            f"Tenure: {tenure} months"
        )

    if (
        customer["InternetService"]
        == "DSL"
    ):

        positive_factors.append(
            "DSL connection"
        )

    if stay_percent > 80:

        positive_factors.append(
            f"High predicted stay probability: "
            f"{stay_percent}%"
        )

    if (
        customer["Contract"]
        == "One year"
    ):

        positive_factors.append(
            "One-year contract"
        )

    if (
        customer["Contract"]
        == "Two year"
    ):

        positive_factors.append(
            "Two-year contract"
        )

    if customer["PaymentMethod"] in [

        "Bank transfer (automatic)",

        "Credit card (automatic)",

    ]:

        positive_factors.append(
            "Automatic payment method"
        )

    if (
        customer["Contract"]
        == "Month-to-month"
    ):

        risk_factors.append(
            "Month-to-month contract"
        )

    if (
        customer["PaymentMethod"]
        == "Electronic check"
    ):

        risk_factors.append(
            "Electronic Check payment"
        )

    if monthly_charges > 80:

        risk_factors.append(
            f"High Monthly Charges: "
            f"${monthly_charges:.2f}"
        )

    if (
        customer["InternetService"]
        == "Fiber optic"
    ):

        risk_factors.append(
            "Fiber optic internet service"
        )

    if tenure < 12:

        risk_factors.append(
            f"Short tenure: {tenure} months"
        )

    if not positive_factors:

        positive_factors.append(
            "No major positive factor identified."
        )

    if not risk_factors:

        risk_factors.append(
            "No major risk factor identified."
        )

    return (
        positive_factors,
        risk_factors
    )


# =========================================================
# BUILD CUSTOMER INPUT
# =========================================================

def build_customer_from_form():

    tenure = int(
        request.form.get(
            "tenure",
            0
        )
    )

    monthly_charges = float(
        request.form.get(
            "MonthlyCharges",
            0
        )
    )

    total_charges = float(
        request.form.get(
            "TotalCharges",
            0
        )
    )

    customer = {

        "gender":
            request.form.get(
                "gender",
                "Male"
            ),

        "SeniorCitizen":
            int(
                request.form.get(
                    "SeniorCitizen",
                    0
                )
            ),

        "Partner":
            request.form.get(
                "Partner",
                "No"
            ),

        "Dependents":
            request.form.get(
                "Dependents",
                "No"
            ),

        "tenure":
            tenure,

        "PhoneService":
            request.form.get(
                "PhoneService",
                "Yes"
            ),

        "MultipleLines":
            request.form.get(
                "MultipleLines",
                "No"
            ),

        "InternetService":
            request.form.get(
                "InternetService",
                "DSL"
            ),

        "OnlineSecurity":
            request.form.get(
                "OnlineSecurity",
                "No"
            ),

        "OnlineBackup":
            request.form.get(
                "OnlineBackup",
                "No"
            ),

        "DeviceProtection":
            request.form.get(
                "DeviceProtection",
                "No"
            ),

        "TechSupport":
            request.form.get(
                "TechSupport",
                "No"
            ),

        "StreamingTV":
            request.form.get(
                "StreamingTV",
                "No"
            ),

        "StreamingMovies":
            request.form.get(
                "StreamingMovies",
                "No"
            ),

        "Contract":
            request.form.get(
                "Contract",
                "Month-to-month"
            ),

        "PaperlessBilling":
            request.form.get(
                "PaperlessBilling",
                "Yes"
            ),

        "PaymentMethod":
            request.form.get(
                "PaymentMethod",
                "Electronic check"
            ),

        "MonthlyCharges":
            monthly_charges,

        "TotalCharges":
            total_charges,
    }

    return customer

# =========================================================
# ALIGN INPUT WITH SAVED PREPROCESSOR
# =========================================================

def prepare_model_input(customer):

    dataset = load_dataset()

    # -----------------------------------------------------
    # Start with one real dataset row.
    # This guarantees that all columns expected by the
    # saved preprocessor exist.
    # -----------------------------------------------------

    if len(dataset) > 0:

        row = dataset.iloc[0].copy()

        # Remove target column
        row = row.drop(
            labels=["Churn"],
            errors="ignore"
        )

        input_df = pd.DataFrame([row])

    else:

        input_df = pd.DataFrame([customer])

    # -----------------------------------------------------
    # IMPORTANT FIX
    #
    # The original Telco dataset contains TotalCharges
    # as a string/object column.
    #
    # We convert the entire input dataframe to object
    # BEFORE assigning form values.
    #
    # This prevents:
    # TypeError: Invalid value '3000.0' for dtype 'str'
    # -----------------------------------------------------

    input_df = input_df.astype(object)

    # -----------------------------------------------------
    # Overwrite dataset values with user input
    # -----------------------------------------------------

    for key, value in customer.items():

        if key in input_df.columns:

            input_df.loc[0, key] = value

    # -----------------------------------------------------
    # Align with the exact columns expected by the
    # saved preprocessor
    # -----------------------------------------------------

    if hasattr(preprocessor, "feature_names_in_"):

        expected_columns = list(
            preprocessor.feature_names_in_
        )

        print("\nExpected model columns:")
        print(expected_columns)

        # Add missing columns
        for column in expected_columns:

            if column not in input_df.columns:

                if column in dataset.columns:

                    input_df[column] = dataset[column].iloc[0]

                elif column == "customerID":

                    input_df[column] = "WEB-INPUT"

                else:

                    raise ValueError(
                        "The saved preprocessor requires "
                        f"column '{column}', but it is not "
                        "available in the application input."
                    )

        # Keep exactly the columns used during training
        input_df = input_df[expected_columns]

    # -----------------------------------------------------
    # Convert numeric columns AFTER assignment
    # -----------------------------------------------------

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:

        if column in input_df.columns:

            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )

    # -----------------------------------------------------
    # Clean string columns
    # -----------------------------------------------------

    string_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    for column in string_columns:

        if column in input_df.columns:

            input_df[column] = (
                input_df[column]
                .astype(str)
                .str.strip()
            )

    # -----------------------------------------------------
    # Final NaN check
    # -----------------------------------------------------

    if input_df.isnull().any().any():

        bad_columns = (
            input_df.columns[
                input_df.isnull().any()
            ].tolist()
        )

        raise ValueError(
            "Missing/invalid values in: "
            + ", ".join(bad_columns)
        )

    # -----------------------------------------------------
    # Debug information
    # -----------------------------------------------------

    print("\n==========================================")
    print("FINAL MODEL INPUT")
    print("==========================================")
    print(input_df)
    print("\nData Types:")
    print(input_df.dtypes)
    print("\nColumns:")
    print(list(input_df.columns))
    print("==========================================\n")

    return input_df

# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # INPUT
        # -------------------------------------------------

        customer = (
            build_customer_from_form()
        )

        tenure = customer[
            "tenure"
        ]

        monthly_charges = customer[
            "MonthlyCharges"
        ]

        total_charges = customer[
            "TotalCharges"
        ]

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if tenure < 0 or tenure > 72:

            raise ValueError(
                "Tenure must be between "
                "0 and 72 months."
            )

        if (
            monthly_charges < 0
            or monthly_charges > 200
        ):

            raise ValueError(
                "Monthly Charges must be "
                "between $0 and $200."
            )

        if total_charges < 0:

            raise ValueError(
                "Total Charges cannot be negative."
            )

        # -------------------------------------------------
        # PREPARE MODEL INPUT
        # -------------------------------------------------

        df = prepare_model_input(
            customer
        )

        print("\n==========================================")
        print("PREDICTION INPUT")
        print("==========================================")
        print(df)
        print("\nColumns:")
        print(list(df.columns))
        print("==========================================\n")

        # -------------------------------------------------
        # TRANSFORM
        # -------------------------------------------------

        start_time = time.perf_counter()

        X = preprocessor.transform(
            df
        )

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if not hasattr(
            model,
            "predict_proba"
        ):

            raise RuntimeError(
                "The saved model does not "
                "support predict_proba()."
            )

        probabilities = (
            model.predict_proba(X)[0]
        )

        if hasattr(
            model,
            "classes_"
        ):

            classes = list(
                model.classes_
            )

            if 1 not in classes:

                raise RuntimeError(
                    "The saved model does not "
                    "contain churn class 1."
                )

            churn_index = (
                classes.index(1)
            )

            probability = float(
                probabilities[churn_index]
            )

        else:

            probability = float(
                probabilities[1]
            )

        # -------------------------------------------------
        # TIME
        # -------------------------------------------------

        prediction_time_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        # -------------------------------------------------
        # PERCENTAGES
        # -------------------------------------------------

        churn_percent = round(
            probability * 100,
            2
        )

        stay_percent = round(
            (1 - probability) * 100,
            2
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:

            result = (
                "❌ Customer is likely to Churn"
            )

            color = "red"

        else:

            result = (
                "✅ Customer is likely to Stay"
            )

            color = "green"
        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = round(
            max(
                churn_percent,
                stay_percent
            ),
            2
        )

        confidence_level = (
            get_confidence_level(
                confidence
            )
        )

        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        risk, badge = get_risk(
            churn_percent
        )

        print("\n========== PREDICTION DEBUG ==========")
        print("Prediction:", result)
        print("Stay Probability:", stay_percent)
        print("Churn Probability:", churn_percent)
        print("======================================\n")
        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        recommendation = (
            build_recommendations(
                customer,
                prediction,
                tenure,
                monthly_charges
            )
        )

        positive_factors, risk_factors = (
            build_factors(
                customer,
                tenure,
                monthly_charges,
                stay_percent
            )
        )

        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        history = get_history()

        if history.empty:

            new_id = 1

        else:

            numeric_ids = pd.to_numeric(
                history["ID"],
                errors="coerce"
            ).dropna()

            if numeric_ids.empty:

                new_id = 1

            else:

                new_id = (
                    int(
                        numeric_ids.max()
                    ) + 1
                )

        new_record = pd.DataFrame(
            {

                "ID":
                    [new_id],

                "Timestamp":
                    [
                        datetime.now().strftime(
                            "%d-%m-%Y %H:%M"
                        )
                    ],

                "Prediction":
                    [result],

                "Probability":
                    [churn_percent],

                "Confidence":
                    [
                        f"{confidence}% "
                        f"({confidence_level})"
                    ],

                "Risk":
                    [risk],
            }
        )

        history = pd.concat(
            [
                history,
                new_record
            ],
            ignore_index=True
        )

        history.to_csv(
            HISTORY_PATH,
            index=False
        )

        history_display = (
            history
            .tail(10)
            .iloc[::-1]
        )

        # -------------------------------------------------
        # PAGE DATA
        # -------------------------------------------------

        page_data = get_page_data()

        page_data.pop(
            "history",
            None
        )

        page_data[
            "prediction_time"
        ] = (
            f"{prediction_time_ms} ms"
        )

        # -------------------------------------------------
        # RENDER
        # -------------------------------------------------

        return render_template(

            "index.html",

            prediction=result,

            probability=churn_percent,

            color=color,

            risk=risk,

            badge=badge,

            confidence=confidence,

            confidence_level=(
                confidence_level
            ),

            stay_probability=(
                stay_percent
            ),

            churn_probability=(
                churn_percent
            ),

            recommendation=(
                recommendation
            ),

            positive_factors=(
                positive_factors
            ),

            risk_factors=(
                risk_factors
            ),

            history=(
                history_display
                .to_dict("records")
            ),

            **page_data
        )

    except ValueError as e:

        print(
            "\nVALIDATION ERROR:",
            repr(e)
        )

        return render_template(
            "index.html",
            error=str(e),
            **get_page_data()
        )

    except Exception as e:

        print("\n==========================================")
        print("PREDICTION ERROR")
        print("==========================================")
        print(
            type(e).__name__,
            ":",
            repr(e)
        )
        print("==========================================\n")

        return render_template(

            "index.html",

            error=(
                "Prediction failed: "
                f"{type(e).__name__}: {e}"
            ),

            **get_page_data()
        )


# =========================================================
# DOWNLOAD HISTORY
# =========================================================

@app.route(
    "/download-history"
)
def download_history():

    if not os.path.exists(
        HISTORY_PATH
    ):

        return (
            "No prediction history available."
        )

    return send_file(

        HISTORY_PATH,

        as_attachment=True,

        download_name=(
            "customer_churn_prediction_history.csv"
        ),

        mimetype="text/csv"
    )


# =========================================================
# FEATURE NAMES
# =========================================================

def get_transformed_feature_names():

    try:

        names = (
            preprocessor
            .get_feature_names_out()
        )

        return [
            str(name)
            for name in names
        ]

    except Exception:

        pass

    try:

        sample = (
            load_dataset()
            .drop(
                columns=["Churn"],
                errors="ignore"
            )
            .head(1)
        )

        transformed = (
            preprocessor.transform(
                sample
            )
        )

        return [
            f"Feature {i + 1}"
            for i in range(
                transformed.shape[1]
            )
        ]

    except Exception:

        return []


# =========================================================
# SHAP / FEATURE IMPORTANCE
# =========================================================

def create_feature_importance_image():

    os.makedirs(
        os.path.dirname(
            SHAP_IMAGE_PATH
        ),
        exist_ok=True
    )

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except Exception as e:

        print(
            "Matplotlib unavailable:",
            repr(e)
        )

        return False

    dataset = (
        load_dataset()
        .drop(
            columns=["Churn"],
            errors="ignore"
        )
    )

    sample_df = dataset.head(500)

    try:

        X_sample = (
            preprocessor.transform(
                sample_df
            )
        )

    except Exception as e:

        print(
            "Could not transform SHAP sample:",
            repr(e)
        )

        return False

    feature_names = (
        get_transformed_feature_names()
    )

    try:

        import shap

        if hasattr(
            model,
            "coef_"
        ):

            explainer = (
                shap.LinearExplainer(
                    model,
                    X_sample
                )
            )

        elif hasattr(
            model,
            "feature_importances_"
        ):

            explainer = (
                shap.TreeExplainer(
                    model
                )
            )

        else:

            explainer = (
                shap.Explainer(
                    model,
                    X_sample
                )
            )

        shap_values = (
            explainer(X_sample)
        )

        values = shap_values.values

        if getattr(
            values,
            "ndim",
            0
        ) == 3:

            values = values[:, :, 1]

        mean_abs = (
            abs(values)
            .mean(axis=0)
        )

        if len(feature_names) != len(
            mean_abs
        ):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(
                    len(mean_abs)
                )
            ]

        importance = pd.Series(
            mean_abs,
            index=feature_names
        ).sort_values(
            ascending=False
        ).head(15)

        plt.figure(
            figsize=(10, 7)
        )

        importance.sort_values().plot(
            kind="barh"
        )

        plt.xlabel(
            "Mean |SHAP value|"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "SHAP Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_IMAGE_PATH,
            dpi=160
        )

        plt.close()

        return True

    except Exception as e:

        print(
            "SHAP generation unavailable:",
            repr(e)
        )

    # -----------------------------------------------------
    # Linear fallback
    # -----------------------------------------------------

    try:

        if hasattr(
            model,
            "coef_"
        ):

            coefficients = (
                model.coef_[0]
            )

            if len(feature_names) != len(
                coefficients
            ):

                feature_names = [
                    f"Feature {i + 1}"
                    for i in range(
                        len(coefficients)
                    )
                ]

            importance = pd.Series(
                abs(coefficients),
                index=feature_names
            ).sort_values(
                ascending=False
            ).head(15)

            plt.figure(
                figsize=(10, 7)
            )

            importance.sort_values().plot(
                kind="barh"
            )

            plt.xlabel(
                "Absolute model coefficient"
            )

            plt.ylabel(
                "Feature"
            )

            plt.title(
                "Feature Importance - Linear Model"
            )

            plt.tight_layout()

            plt.savefig(
                SHAP_IMAGE_PATH,
                dpi=160
            )

            plt.close()

            return True

    except Exception as e:

        print(
            "Feature importance fallback failed:",
            repr(e)
        )

    return False


# =========================================================
# FEATURE IMPORTANCE PAGE
# =========================================================

@app.route(
    "/feature-importance"
)
def feature_importance():

    if not os.path.exists(
        SHAP_IMAGE_PATH
    ):

        create_feature_importance_image()

    page_data = get_page_data()

    return render_template(

        "feature_importance.html",

        top_features=(
            page_data[
                "top_features"
            ]
        )
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("")
    print("==========================================")
    print(" CUSTOMER CHURN PREDICTION SYSTEM")
    print("==========================================")
    print(
        "Project:",
        BASE_DIR
    )
    print(
        "Dataset:",
        DATASET_PATH
    )
    print(
        "Model:",
        MODEL_PATH
    )
    print(
        "Preprocessor:",
        PREPROCESSOR_PATH
    )
    print("==========================================")
    print("")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )