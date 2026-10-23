from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "home-credit-default-risk" / "application_train.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "model"
MODEL_PATH = MODEL_DIR / "lightgbm_credit_model.joblib"


def main():
    print("Loading training data...")
    data = pd.read_csv(DATA_PATH)

    y = data["TARGET"]
    X = data.drop(columns=["TARGET", "SK_ID_CURR"]).copy()

    X["AGE_YEARS"] = -X["DAYS_BIRTH"] / 365.25
    X["EMPLOYED_YEARS"] = (
        X["DAYS_EMPLOYED"].replace(365243, float("nan")).abs() / 365.25
    )
    X["CREDIT_INCOME_RATIO"] = (
        X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"].replace(0, float("nan"))
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=50000,
        random_state=42,
        stratify=y_train,
    )

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=100,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])

    print(f"Training on {len(X_sample):,} applicants...")
    model.fit(X_sample, y_sample)

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nValidation results")
    print(f"ROC-AUC:           {roc_auc_score(y_valid, probabilities):.4f}")
    print(f"Average precision: {average_precision_score(y_valid, probabilities):.4f}")
    print(f"Precision:         {precision_score(y_valid, predictions, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(y_valid, predictions, zero_division=0):.4f}")
    print(f"Accuracy:          {accuracy_score(y_valid, predictions):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
