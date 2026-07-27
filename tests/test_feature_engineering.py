import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features


def test_feature_engineering():

    df = pd.read_csv("data/raw/telco_churn.csv")

    df = preprocess_data(df)

    df = create_features(df)

    expected_features = [
        "AvgMonthlySpend",
        "IsNewCustomer",
        "IsMonthToMonth",
        "HasFiberOptic",
        "ServiceCount"
    ]

    for feature in expected_features:
        assert feature in df.columns

    print("Feature engineering test passed.")