
import pandas as pd

#  This function creates the same three engineered
#  features for both training and future
#  customer data.
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    df["EMPLOYED_YEARS"] = (
        df["DAYS_EMPLOYED"].replace(365243, float("nan")).abs() / 365.25
    )

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, float("nan"))
    )

    return df
