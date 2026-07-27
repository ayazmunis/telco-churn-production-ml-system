"""
drift_check.py

Simple data quality and drift checks.
"""

import pandas as pd

def check_missing_values(df):
    """
    Check for missing values in the incoming dataset.
    """

    print("\nChecking Missing Values...")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("✓ No missing values detected.")
    else:
        print("⚠ Missing values detected:")
        print(missing[missing > 0])

def check_schema(train_df, new_df):
    """
    Ensure the incoming data has the expected schema.
    """

    print("\nChecking Schema...")

    expected_columns = set(train_df.columns)
    incoming_columns = set(new_df.columns)

    missing_columns = expected_columns - incoming_columns
    extra_columns = incoming_columns - expected_columns

    if not missing_columns and not extra_columns:
        print("✓ Schema validation passed.")
    else:

        if missing_columns:
            print(f"⚠ Missing Columns: {missing_columns}")

        if extra_columns:
            print(f"⚠ Unexpected Columns: {extra_columns}")

def check_feature_drift(train_df, new_df):
    """
    Compare MonthlyCharges distribution.
    """

    print("\nChecking Feature Drift...")

    train_mean = train_df["MonthlyCharges"].mean()
    new_mean = new_df["MonthlyCharges"].mean()

    difference = abs(train_mean - new_mean)

    print(f"Training Mean : {train_mean:.2f}")
    print(f"New Data Mean : {new_mean:.2f}")
    print(f"Difference    : {difference:.2f}")

    drift_detected = difference > 10

    if drift_detected:
        print("Potential drift detected.")
    else:
        print("No significant drift detected.")

    return drift_detected

def check_monthly_charge_drift(train_df, new_df):
    """
    Compare MonthlyCharges mean between
    training data and the new batch.
    """

    train_mean = train_df["MonthlyCharges"].mean()

    new_mean = new_df["MonthlyCharges"].mean()

    difference = abs(train_mean - new_mean)

    print("\nMonthlyCharges Drift Check")

    print(f"Training Mean : {train_mean:.2f}")
    print(f"New Data Mean : {new_mean:.2f}")
    print(f"Difference    : {difference:.2f}")

    if difference > 10:

        print("\nWARNING: Possible data drift detected!")

    else:

        print("\nNo significant drift detected.")

def main():

    train_df = pd.read_csv(
        "data/raw/telco_churn.csv"
    )

    new_df = pd.read_csv(
        "data/new_data/new_telco_data.csv"
    )

    check_missing_values(new_df)

    check_schema(
        train_df,
        new_df
    )

    check_feature_drift(
        train_df,
        new_df
    )


if __name__ == "__main__":
    main()