"""
ingestion.py

Simple batch data ingestion script.
"""

import os
import pandas as pd
from datetime import datetime
from config import config


def ingest_data(existing_file, new_file, output_file):
    """
    Read new data, merge with existing training data,
    and save the updated dataset.
    """
    # Load datasets
    existing_df = pd.read_csv(existing_file)
    new_df = pd.read_csv(new_file)

    # Merge datasets
    updated_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    # Save merged dataset
    updated_df.to_csv(output_file, index=False)

    # ==========================================
    # Save ingestion log
    # ==========================================

    os.makedirs("artifacts/logs", exist_ok=True)

    log_file = "artifacts/logs/ingestion.log"

    with open(log_file, "a") as file:
        file.write(
            f"{datetime.now()} | "
            f"Rows Added: {len(new_df)} | "
            f"Total Rows: {len(updated_df)}\n"
        )

    print("\n========== Data Ingestion ==========")

    print(f"Timestamp : {datetime.now()}")

    print(f"Rows Added : {len(new_df)}")

    print(f"Total Rows : {len(updated_df)}")

    print("====================================")

if __name__ == "__main__":

    ingest_data(
        existing_file=config["data"]["raw_data"],
        new_file=config["data"]["new_data"],
        output_file=config["data"]["processed_data"]
    )