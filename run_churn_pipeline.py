from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor, load_data, prepare_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main() -> None:
    df = load_data(DATA_PATH)
    X, y, numeric_features, categorical_features = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = create_preprocessor(numeric_features, categorical_features)
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"ROC AUC: {auc:.4f}")

    model_path = MODELS_DIR / "churn_model_v1.pkl"
    with model_path.open("wb") as fh:
        import pickle

        pickle.dump(pipeline, fh)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
