import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocessing():

    df = pd.read_csv("data/raw/telco_churn.csv")

    cleaned_df = preprocess_data(df)

    assert cleaned_df.isnull().sum().sum() == 0

    print("Preprocessing test passed.")