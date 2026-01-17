import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from dataset_encoder import DatasetEncoder

def mase(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    mae_model = np.mean(np.abs(y_test - y_pred))

    naive_guess = np.median(y_test)
    mae_naive = np.mean(np.abs(y_test - naive_guess))

    return mae_model / mae_naive


def main():
    encoded_dataset = pd.read_csv(Path(__file__).resolve().parents[1] / "encoded_datasets/encoded_time_to_convergence.csv")
    X = encoded_dataset.drop(columns=["time_to_convergence"], errors="ignore")
    y = encoded_dataset["time_to_convergence"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    y_test_secs = np.expm1(y_test)
    preds = regr.predict(X_test)
    preds_secs = np.expm1(preds)

    print("Mean absolute error: ", mean_absolute_error(y_test_secs, preds_secs))
    print("R2: ", r2_score(y_test, preds))
    print("Median absolute error: ", median_absolute_error(y_test_secs, preds_secs))
    print("Mean absolute scaled error: ", mase(y_test_secs, preds_secs))

    results = X_test.copy()

    results["Actual_Log"] = y_test
    results["Predicted_Log"] = preds

    results["Actual_Time"] = np.expm1(results["Actual_Log"])
    results["Predicted_Time"] = np.expm1(results["Predicted_Log"])

    results["Error_Seconds"] = results["Predicted_Time"] - results["Actual_Time"]

    display_cols = ["Actual_Time", "Predicted_Time", "Error_Seconds"]
    results.to_csv("prediction_convergence_analysis.csv", index=False)


if __name__ == "__main__":
    main()
