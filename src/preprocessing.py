from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset_path(file_path):
    """Resolve a dataset path relative to the project root."""
    path = Path(file_path)
    if path.is_absolute():
        return path

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    return path


def load_data(file_path):
    """Load the Telco customer churn dataset."""
    data_path = resolve_dataset_path(file_path)
    df = pd.read_csv(data_path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def prepare_data(df):
    """Prepare features and target for machine learning."""
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "Churn" not in df.columns:
        raise ValueError("Dataset is missing the 'Churn' target column.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    if "tenure" in X.columns:
        X = X.copy()
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        )
        categorical_features = list(dict.fromkeys(categorical_features + ["tenure_group"]))

    return X, y, numeric_features, categorical_features


def create_preprocessor(numeric_features, categorical_features):
    """Create the preprocessing pipeline."""
    transformers = []

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    if not transformers:
        raise ValueError("At least one feature column must be provided for preprocessing.")

    return ColumnTransformer(transformers=transformers)


__all__ = ["load_data", "prepare_data", "create_preprocessor"]
