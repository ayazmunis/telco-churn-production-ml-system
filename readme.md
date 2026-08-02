# Telco Churn Prediction - Mini Production ML System

## Project Overview

This project implements a miniature production-ready Machine Learning system for predicting customer churn using the IBM Telco Customer Churn dataset.

Unlike a traditional machine learning notebook, this project demonstrates the complete machine learning lifecycle including data preprocessing, feature engineering, model training, model evaluation, deployment through a REST API, data ingestion, monitoring, drift detection, retraining logic, and latency measurement.

The primary objective is to simulate how a machine learning model would be deployed and maintained in a production environment while following good software engineering practices.

---

## Objectives

- Predict whether a customer is likely to churn.
- Build a repeatable machine learning training pipeline.
- Compare baseline and candidate models.
- Deploy the selected model using FastAPI.
- Simulate batch data ingestion.
- Monitor data quality and feature drift.
- Define retraining conditions.
- Measure API inference latency.

---

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

**Problem Type:** Binary Classification

**Target Variable:**

- Churn
    - 0 = Customer stays
    - 1 = Customer churns

The dataset contains customer demographics, subscribed services, billing information, and contract details.

---

## Project Architecture

The system consists of the following stages:

1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Model Selection
6. Model Deployment
7. Batch Data Ingestion
8. Monitoring & Drift Detection
9. Retraining Decision
10. API Performance Measurement

---

## Project Structure

```text
artifacts/
│
├── evaluation/
├── logs/
│
config/
│
data/
│
├── raw/
├── new_data/
└── processed/
│
models/
│
src/
│
tests/
```

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Requests
- PyYAML

---

## Installation

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Training Pipeline

```bash
python src/train.py
```

The training pipeline performs the following steps:

- Loads the dataset
- Cleans the data
- Performs feature engineering
- Splits train/test data
- Trains baseline and candidate models
- Evaluates both models
- Applies promotion guardrails
- Saves the selected model
- Saves evaluation metrics

---

## Running the Prediction API

Start the FastAPI server

```bash
uvicorn src.app:app --reload
```

Swagger documentation is available at

```
http://127.0.0.1:8000/docs
```

Prediction endpoint

```
POST /predict
```

The API returns

- prediction
- churn_probability
- model_version

---

## Running Data Ingestion

```bash
python src/ingestion.py
```

The ingestion pipeline:

- Reads new customer records
- Appends them to the processed dataset
- Logs ingestion metadata
- Stores ingestion history

---

## Monitoring

Run

```bash
python src/drift_check.py
```

The monitoring module performs:

- Missing value detection
- Schema validation
- Feature drift detection

---

## Retraining Logic

Run

```bash
python src/retraining.py
```

Retraining is triggered if:

- Feature drift exceeds threshold
- Large amounts of new data are collected
- Model performance drops below acceptable limits

---

## Latency Testing

Run

```bash
python src/latency_test.py
```

The latency test reports:

- Successful requests
- Failed requests
- Average latency
- Minimum latency
- Maximum latency
- P95 latency

---

## Model Performance

Two models were evaluated:

- Baseline Logistic Regression
- Candidate Random Forest

The deployment decision is based on production guardrails using ROC-AUC rather than simply selecting the highest-performing model.

---

## Future Improvements

Possible enhancements include:

- Docker containerization
- CI/CD pipeline
- Automated retraining scheduler
- Feature store integration
- Model registry
- Cloud deployment
- Advanced drift detection techniques
- Comprehensive unit and integration testing

---

## Author

Ayaz Munis

Machine Learning Model Engineering Project