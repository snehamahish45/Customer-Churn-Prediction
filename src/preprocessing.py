import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load dataset from CSV
    """

    df = pd.read_csv(file_path)

    return df


def clean_data(df):
    """
    Clean the dataset
    """

    # Remove customerID
    if "customerID" in df.columns:

        df.drop(
            "customerID",
            axis=1,
            inplace=True
        )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df.dropna(
        inplace=True
    )

    return df


def split_features_target(df):
    """
    Separate features and target
    """

    X = df.drop(
        "Churn",
        axis=1
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


def build_preprocessor(X):

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                StandardScaler(),
                numerical_columns
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )

        ]

    )

    return preprocessor


def prepare_data(df):

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    preprocessor = build_preprocessor(X)

    X_train = preprocessor.fit_transform(
        X_train
    )

    X_test = preprocessor.transform(
        X_test
    )

    # =====================================================
    # SAVE PREPROCESSOR
    # =====================================================

    # Get Customer-Churn-Prediction project root
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # Create models directory
    models_dir = os.path.join(
        project_root,
        "models"
    )

    os.makedirs(
        models_dir,
        exist_ok=True
    )

    # Preprocessor file path
    preprocessor_path = os.path.join(
        models_dir,
        "preprocessor.pkl"
    )

    # Save preprocessor
    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print(
        f"Preprocessor saved successfully: {preprocessor_path}"
    )

    return (

        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor

    )