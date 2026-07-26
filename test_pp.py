import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features

df = pd.read_csv("data/raw/telco_churn.csv")

df = preprocess_data(df)
df = create_features(df)

print(
    df[
        [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "ServiceCount"
        ]
    ].head(10)
)