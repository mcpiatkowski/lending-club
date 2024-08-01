"""Data processing module."""

__all__ = ["execute_processing"]

from collections import namedtuple

import numpy as np
import pandas as pd

from .logger import log

Mapping = namedtuple("Mapping", ["emp_length_map", "grade_map", "loan_status_map"])

MAPPING = Mapping(
    grade_map={"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
    loan_status_map={"Fully Paid": 0, "Charged Off": 1},
    emp_length_map={
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        np.nan: 0,
    },
)


def filter_loan_status(loans: pd.DataFrame) -> pd.DataFrame:
    """Filter loan status.

    There is some volume that has different statuses that we don't need for the inference.
    """
    log.info("Filtering loan status...")

    return loans[loans["loan_status"].isin(["Fully Paid", "Charged Off"])]


def filter_missing_data(loans: pd.DataFrame) -> pd.DataFrame:
    """Filter missing data.

    Drop columns with more than half of the values missing.
    """
    log.info("Filtering missing data...")

    return loans.dropna(thresh=int(len(loans) / 2), axis="columns")


def remove_unnecessary_columns(loans: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary columns.

    For the column description, please refer to the dictionary.
    """
    log.info("Removing unnecessary columns...")

    return loans.drop(
        columns=[
            "id",
            "collection_recovery_fee",  # More than 90% zero
            "total_rec_late_fee",  # More than 90% zero
            "debt_settlement_flag",
            "url",
            "desc",
            "total_pymnt",  # Leaks data from the future
            "total_pymnt_inv",  # Leaks data from the future
            "recoveries",
            "zip_code",
            "sub_grade",
            "emp_title",
            "int_rate",  # Information contained in grade column.
            # Leaks data from the future
            "last_fico_range_high",
            "last_fico_range_low",
            "issue_d",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "last_pymnt_d",
            "last_pymnt_amnt",
            "funded_amnt",
            "funded_amnt_inv",
            # One value is overwhelming
            "pub_rec_bankruptcies",
            "pub_rec",
            "delinq_2yrs",
            # Try some feature engineering on those below
            "earliest_cr_line",
            "last_credit_pull_d",
            # Categorical too many unique values
            "addr_state",
            "title",
        ]
    )


def filter_non_unique_values(loans: pd.DataFrame) -> pd.DataFrame:
    """Filter non unique values."""
    log.info("Filtering non unique values...")

    return loans.loc[:, loans.nunique() != 1]


def map_categorical(loans: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """Map categorical columns."""
    log.info("Mapping categorical columns...")

    return (
        loans.assign(grade=loans["grade"].map(mapping.grade_map))
        .assign(emp_length=loans["emp_length"].map(mapping.emp_length_map))
        .assign(loan_status=loans["loan_status"].map(mapping.loan_status_map))
    )


def apply_transformations(loans: pd.DataFrame) -> pd.DataFrame:
    """Apply string transformation."""
    log.info("Applying transformation...")

    return (
        loans.assign(term=loans["term"].str.strip("months"))
        .assign(revol_util=pd.to_numeric(loans["revol_util"].str.strip("%")))
        .assign(home_ownership=loans["home_ownership"].replace("NONE", "OTHER"))
        .assign(fico_avg=loans[["fico_range_low", "fico_range_high"]].mean(axis="columns"))
        .drop(columns=["fico_range_low", "fico_range_high"])
    )


def create_dummies(loans: pd.DataFrame) -> pd.DataFrame:
    """Create dummies."""
    log.info("Creating dummies...")
    columns = ["home_ownership", "verification_status", "purpose", "term"]

    return pd.concat([loans, pd.get_dummies(loans[columns], dtype=int)], axis=1).drop(columns=columns).dropna()


def handle_outliers(loans: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers."""
    log.info("Handling outliers...")
    for feature in loans.select_dtypes(include="number"):
        loans[feature] = loans[feature].clip(lower=loans[feature].quantile(0.05), upper=loans[feature].quantile(0.95))

    return loans


def to_upper(loans: pd.DataFrame) -> pd.DataFrame:
    """Convert to upper case."""
    log.info("Converting to upper case...")
    loans.columns = [col.replace(" ", "").upper() for col in loans.columns]

    return loans


def execute_processing(loans: pd.DataFrame) -> pd.DataFrame:
    """Execute processing."""
    log.info("Processing data...")

    return (
        filter_loan_status(loans)
        .pipe(filter_missing_data)
        .pipe(filter_non_unique_values)
        .pipe(remove_unnecessary_columns)
        .pipe(map_categorical, mapping=MAPPING)
        .pipe(apply_transformations)
        .pipe(handle_outliers)
        .pipe(create_dummies)
        .pipe(to_upper)
    )
