"""
feature_engineering.py

Contains all feature engineering logic.
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features.
    """

    # Create a copy of the dataframe
    df = df.copy()

    # ==========================================
    # Spending Features
    # ==========================================

    # Average amount spent per month
    df["AvgMonthlySpend"] = 0.0

    mask = df["tenure"] > 0

    df.loc[mask, "AvgMonthlySpend"] = (
        df.loc[mask, "TotalCharges"] /
        df.loc[mask, "tenure"]
    )

    # ==========================================
    # Customer Lifecycle Features
    # ==========================================

    # Binary feature indicating whether the customer is relatively new
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)

    # ==========================================
    # Contract Features
    # ==========================================

    # Binary feature indicating month-to-month contracts
    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    # ==========================================
    # Internet Features
    # ==========================================   

    # Binary feature indicating Fiber Optic internet users
    df["HasFiberOptic"] = (
        df["InternetService"] == "Fiber optic"
        ).astype(int)
    
    # ==========================================
    # Customer Engagement Features
    # ==========================================    
    
    # List of service-related columns
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    # Count the number of subscribed services
    df["ServiceCount"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    return df