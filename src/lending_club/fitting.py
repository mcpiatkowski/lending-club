"""Fitting module."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .logger import log


def train_test_split(loans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    log.info("Split data into train and test sets...")

    x = loans.drop(columns="LOAN_STATUS").copy()
    y = loans["LOAN_STATUS"].copy()

    shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_index, test_index = next(shuffle.split(x, y))

    return x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]


def create_model_pipeline() -> Pipeline:
    """Creates model pipeline."""
    log.info("Creating model pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                [
                    "LOAN_AMNT",
                    "INSTALLMENT",
                    "ANNUAL_INC",
                    "DTI",
                    "FICO_AVG",
                    "INQ_LAST_6MTHS",
                    "OPEN_ACC",
                    "REVOL_BAL",
                    "REVOL_UTIL",
                    "TOTAL_ACC",
                ],
            )
        ],
        remainder="passthrough",
        force_int_remainder_cols=False,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("resampling", SMOTE(sampling_strategy=0.6)),
            ("classification", LogisticRegression(max_iter=1000)),
        ]
    )
