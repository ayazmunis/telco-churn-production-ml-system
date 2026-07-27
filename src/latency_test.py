"""
latency_test.py

Measure inference latency of the FastAPI prediction service.
"""

import time
import statistics

import requests

NUM_REQUESTS = 100

URL = "http://127.0.0.1:8000/predict"

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

def measure_latency(num_requests=100):
    """
    Measure API latency by sending multiple requests to the prediction API.
    """

    latencies = []

    successful_requests = 0
    failed_requests = 0

    print(f"\nSending {num_requests} requests...")

    for _ in range(num_requests):

        start = time.perf_counter()

        response = requests.post(
            URL,
            json=sample_customer
        )

        end = time.perf_counter()

        latency = (end - start) * 1000

        if response.status_code == 200:

            successful_requests += 1
            latencies.append(latency)

        else:

            failed_requests += 1

            print(
                f"Request failed with status code: {response.status_code}"
            )

    return (
        latencies,
        successful_requests,
        failed_requests
    )


def print_results(
    latencies,
    successful_requests,
    failed_requests
):
    """
    Print latency statistics.
    """

    print("\n========== Latency Report ==========")

    print(f"Total Requests      : {successful_requests + failed_requests}")
    print(f"Successful Requests : {successful_requests}")
    print(f"Failed Requests     : {failed_requests}")

    if not latencies:
        print("\nNo successful requests to calculate latency.")
        return

    print(f"\nAverage Latency : {statistics.mean(latencies):.2f} ms")
    print(f"Minimum Latency : {min(latencies):.2f} ms")
    print(f"Maximum Latency : {max(latencies):.2f} ms")

    p95 = sorted(latencies)[
        int(0.95 * len(latencies))
    ]

    print(f"P95 Latency     : {p95:.2f} ms")

    print("====================================")


def main():

    latencies, successful_requests, failed_requests = measure_latency(
        num_requests=NUM_REQUESTS
    )

    print_results(
        latencies,
        successful_requests,
        failed_requests
    )


if __name__ == "__main__":
    main()