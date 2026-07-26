"""
preprocessing.py

Contains all data cleaning and preprocessing logic.

This module is shared between training and inference to ensure
consistent preprocessing and avoid training-serving skew.
"""

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco Customer Churn dataset.
    """

    # Create a copy so the original dataframe is not modified
    df = df.copy()

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Remove leading/trailing spaces from string values
    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:
        df[col] = df[col].str.strip()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Store the original number of rows
    original_rows = len(df)

    # Remove rows containing missing values
    df = df.dropna()

    # Print preprocessing summary
    print("Preprocessing completed.")
    print(f"Rows before cleaning : {original_rows}")
    print(f"Rows after cleaning  : {len(df)}")
    print(f"Rows removed         : {original_rows - len(df)}")

    # Convert target column to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df