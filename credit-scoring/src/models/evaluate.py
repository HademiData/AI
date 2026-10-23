
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
RESULTS_DIR = PROJECT_ROOT / "results/model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
data = pd.read_csv(DATA_DIR / "application_train.csv")

X = data.drop(columns=["TARGET", "SK_ID_CURR"])
y = data["TARGET"]

X["AGE_YEARS"] = -X["DAYS_BIRTH"] / 365.25
X["EMPLOYED_YEARS"] = (
    X["DAYS_EMPLOYED"].replace(365243, float("nan")).abs() / 365.25
)
X["CREDIT_INCOME_RATIO"] = (
    X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"].replace(0, float("nan"))
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

sizes = [10000, 20000, 30000, 40000, 50000]
train_scores, valid_scores = [], []

for size in sizes:
    X_part, _, y_part, _ = train_test_split(
        X_train,
        y_train,
        train_size=size,
        random_state=42,
        stratify=y_train,
    )

    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
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

    print(f"Training with {size:,} rows...")
    model.fit(X_part, y_part)

    train_auc = roc_auc_score(y_part, model.predict_proba(X_part)[:, 1])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

    train_scores.append(train_auc)
    valid_scores.append(valid_auc)

    print(
        f"Training AUC: {train_auc:.4f} | "
        f"Validation AUC: {valid_auc:.4f}"
    )

plt.figure(figsize=(8, 5))
plt.plot(sizes, train_scores, marker="o", label="Training AUC")
plt.plot(sizes, valid_scores, marker="o", label="Validation AUC")
plt.xlabel("Training rows")
plt.ylabel("ROC-AUC")
plt.title("LightGBM Learning Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = RESULTS_DIR / "learning_curve.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print("Saved:", plot_path)
