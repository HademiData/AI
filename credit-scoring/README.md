# CreditLens: Credit Scoring and Risk Analysis

## Overview

CreditLens is a machine learning project that estimates the probability of recorded payment difficulties for loan applicants using the Home Credit Default Risk dataset.

It combines exploratory data analysis, feature engineering, LightGBM classification, model evaluation, SHAP explanations, interactive Plotly visualizations, and a Streamlit dashboard.

This is an educational decision-support project. Predictions are estimates, not guarantees of future repayment behavior, and should not be used as the sole basis for lending decisions.

## Model Performance

- **Algorithm:** LightGBM classifier
- **Validation ROC-AUC:** 0.7513
- **Validation average precision:** 0.2388
- **Training sample:** 50,000 applicants
- **Validation strategy:** Stratified 80/20 train-validation split
- **Positive class:** TARGET = 1, indicating recorded payment difficulties

Accuracy alone is insufficient because only approximately 8.07% of training applicants belong to the positive class. ROC-AUC, average precision, precision, and recall provide a more informative assessment.

## Features

- Exploratory data analysis and missing-value analysis
- Feature engineering and preprocessing
- Logistic Regression baseline and LightGBM model
- Training and validation learning curve
- Kaggle submission predictions
- Global feature importance and SHAP explanations
- Customer-level risk predictions
- SHAP waterfall plots and force plots
- Interactive Plotly comparisons against the applicant population
- Streamlit dashboard for portfolio overview, customer assessment, and model explainability

## Project Structure

```text
credit-scoring/
├── home-credit-default-risk/
│   ├── application_train.csv
│   ├── application_test.csv
│   └── supporting dataset files
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── explain_customer.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── visualization/
│   │   └── client_report.py
│   └── dashboard/
│       ├── app.py
│       ├── components.py
│       ├── styles.py
│       └── pages/
├── results/
│   ├── model/
│   ├── clients_outputs/
│   └── submission.csv
├── requirements.txt
├── username.txt
└── README.md
```

## Setup

Python 3.12 is the project's tested environment.

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure the Home Credit dataset files are in `home-credit-default-risk/`.

## Run the Project

### 1. Explore the data

Open the notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. Train the model

```bash
python src/models/train.py
```

The script saves the trained model to:

`results/model/lightgbm_credit_model.joblib`

The current training script uses a stratified 50,000-row sample and a fixed random seed.

### 3. Generate the learning curve

```bash
python src/models/evaluate.py
```

The learning curve is saved to:

`results/model/learning_curve.png`

This script trains models on increasing sample sizes to compare training and validation ROC-AUC.

### 4. Generate Kaggle predictions

```bash
python src/models/predict.py
```

The submission file is saved to:

`results/submission.csv`

The test dataset does not contain TARGET labels, so its true performance cannot be evaluated locally.

### 5. Explain a customer

For a test customer:

```bash
python src/models/explain_customer.py 100001 --dataset test
```

For a training customer:

```bash
python src/models/explain_customer.py 100002 --dataset train
```

The script reports the predicted risk and saves SHAP explanations under `results/clients_outputs/`.

### 6. Generate customer comparison reports

```bash
python src/visualization/client_report.py
```

This generates interactive HTML reports for the configured training and test examples.

### 7. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard includes portfolio overview, customer assessment, and model explainability pages.

## Interpretability

SHAP explanations show how model features contribute to predictions. Global explanations summarize feature influence across a sample, while local explanations describe an individual prediction.

Feature contributions describe model behavior, not causation. They should not be interpreted as proof that a particular applicant characteristic causes payment difficulties.

## Limitations and Future Improvements

- The current model is trained on a sample rather than the full training dataset.
- The model may overfit and may perform differently across applicant groups.
- The 0.5 classification threshold is a demonstration choice, not a validated lending policy.
- Predicted probabilities have not necessarily been calibrated to observed default frequencies.
- Early stopping is not currently implemented.
- Further work could include probability calibration, threshold analysis, fairness evaluation, broader model comparisons, and training on more records.
- The test dataset has no labels, so its actual predictive performance cannot be measured locally.

## Project Identity

The 01Edu username is recorded in `username.txt`.
