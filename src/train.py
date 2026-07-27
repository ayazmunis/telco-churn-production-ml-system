"""
train.py

Main training pipeline for the Telco Customer Churn Prediction System.
"""

# ==========================================
# Imports
# ==========================================

import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import preprocess_data
from feature_engineering import create_features
from config import config

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Evaluate a trained classification model.
    """

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 50}")

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")

def load_data():
    """
    Load and prepare the dataset.
    """

    print("Loading dataset...")

    df = pd.read_csv(config["data"]["raw_data"])

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    print("\nPreprocessing data...")
    df = preprocess_data(df)

    print("\nCreating engineered features...")
    df = create_features(df)

    print("Feature engineering complete.")

    return df

def prepare_data(df):
    """
    Separate features and target.
    """

    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"]

    return X, y

def build_preprocessor(X):
    """
    Create preprocessing pipeline for categorical features.
    """

    categorical_columns = (
        X.select_dtypes(include="object")
        .columns
        .tolist()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    return preprocessor

def build_baseline_pipeline(preprocessor):

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

def build_candidate_pipeline(preprocessor):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )

def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    """

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics for a classification model.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

def train_and_evaluate(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name):
    """
    Train a pipeline and evaluate its performance.
    """

    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(
        y_test,
        predictions,
        probabilities
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    for metric, value in metrics.items():
        print(f"{metric.capitalize():10}: {value:.4f}")

    return {
        "name": model_name,
        "pipeline": pipeline,
        "metrics": metrics
    }

def select_best_model(
    baseline_results,
    candidate_results
):
    """
    Select the best model based on ROC-AUC.
    """

    print("\n" + "=" * 50)
    print("Model Promotion Decision")
    print("=" * 50)

    if (
        candidate_results["metrics"]["auc"]
        >
        baseline_results["metrics"]["auc"]
    ):

        print(
            f"Promoting {candidate_results['name']}"
        )

        return (
            candidate_results["pipeline"],
            candidate_results["name"]
        )

    print(
        f"Keeping {baseline_results['name']}"
    )

    return (
        baseline_results["pipeline"],
        baseline_results["name"]
    )

def save_model(model, model_name):
    """
    Save the selected model pipeline.
    """

    os.makedirs("models", exist_ok=True)

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    filepath = os.path.join("models", filename)

    joblib.dump(model, filepath)

    print(f"\nModel saved successfully!")
    print(f"Location: {filepath}")

def save_evaluation_report(
    baseline_results,
    candidate_results
):
    """
    Save model evaluation metrics to a JSON file.
    """

    os.makedirs("artifacts", exist_ok=True)

    report = {
        "baseline": {
            "model": baseline_results["name"],
            "metrics": baseline_results["metrics"]
        },
        "candidate": {
            "model": candidate_results["name"],
            "metrics": candidate_results["metrics"]
        }
    }

    os.makedirs("artifacts/evaluation", exist_ok=True)

    filepath = "artifacts/evaluation/evaluation.json"

    with open(filepath, "w") as file:
        json.dump(report, file, indent=4)

    print(f"Evaluation report saved: {filepath}")


def main():
    """
    Main training pipeline.
    """
    # Load and prepare dataset
    df = load_data()

    # Separate features and target
    X, y = prepare_data(df)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create encoder
    preprocessor = build_preprocessor(X)

    # Build models
    baseline_pipeline = build_baseline_pipeline(preprocessor)
    candidate_pipeline = build_candidate_pipeline(preprocessor)

    baseline_results = train_and_evaluate(
        baseline_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        "Baseline Logistic Regression"
    )

    candidate_results = train_and_evaluate(
        candidate_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        "Candidate Random Forest"
    )

    best_model, best_model_name = select_best_model(
        baseline_results,
        candidate_results
    )

    save_model(
        best_model,
        best_model_name
    )

    save_evaluation_report(
        baseline_results,
        candidate_results
    )

if __name__ == "__main__":
    main()