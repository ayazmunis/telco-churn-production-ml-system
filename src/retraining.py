"""
retraining.py

Simple retraining trigger logic.
"""

import pandas as pd

from drift_check import check_feature_drift
from config import config

def should_retrain(
    drift_detected,
    new_rows,
    auc_drop
):
    """
    Decide whether the model should be retrained.
    """

    if drift_detected:
        return True, "Feature drift detected."

    if new_rows >= 1000:
        return True, "Sufficient new data collected."

    if auc_drop >= 0.05:
        return True, "Model performance has degraded."

    return False, "No retraining required."

def main():

    print("\nChecking Retraining Conditions...")

    train_df = pd.read_csv(
        config["data"]["raw_data"]
    )

    new_df = pd.read_csv(
        config["data"]["new_data"]
    )

    drift_detected = check_feature_drift(
        train_df,
        new_df
    )

    new_rows = len(new_df)

    # Simulated AUC drop
    auc_drop = 0.00

    retrain, reason = should_retrain(
        drift_detected,
        new_rows,
        auc_drop
    )

    print("\n========== Retraining Decision ==========")

    if retrain:
        print("Retraining Required")
    else:
        print("No Retraining Required")

    print(f"Reason: {reason}")

    print("=========================================")

if __name__ == "__main__":
    main()