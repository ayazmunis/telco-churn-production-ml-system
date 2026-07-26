import pandas as pd
from src.preprocessing import preprocess_data

df = pd.read_csv("data/raw/telco_churn.csv")

clean_df = preprocess_data(df)

print(clean_df.head())
print(clean_df.info())