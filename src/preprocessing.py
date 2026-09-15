import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(file_path):
    """Load the Telco customer churn dataset."""
    df = pd.read_csv(file_path)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Fill missing TotalCharges values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def prepare_data(df):
    """Prepare features and target for machine learning."""

    # Drop customer identifier
    df = df.drop("customerID", axis=1)

    # Convert target to numerical values
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include="object"
    ).columns.tolist()

    # Add engineered feature
    X["tenure_group"] = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )

    categorical_features.append("tenure_group")

    return X, y, numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """Create the preprocessing pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor
from src.preprocessing import load_data, prepare_data, create_preprocessor
