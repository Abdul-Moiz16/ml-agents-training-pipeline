import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

def mase(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    mae_model = np.mean(np.abs(y_test - y_pred))

    naive_guess = np.median(y_test)
    mae_naive = np.mean(np.abs(y_test - naive_guess))

    return mae_model / mae_naive


def main():
    encoded_dataset = pd.read_csv(Path(__file__).resolve().parents[1] / "encoded_datasets/encoded_avg_ram_usage.csv")
    X = encoded_dataset.drop(columns=["avg_ram_usage"], errors="ignore")
    y = encoded_dataset["avg_ram_usage"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    preds = regr.predict(X_test)

    print("Mean absolute error: ", mean_absolute_error(y_test, preds))
    print("R2: ", r2_score(y_test, preds))
    print("Median absolute error: ", median_absolute_error(y_test, preds))
    print("Mean absolute scaled error: ", mase(y_test, preds))

    results = X_test.copy()

    results["Actual_mb"] = y_test
    results["Predicted_mb"] = preds

    results["Error_mb"] = results["Predicted_mb"] - results["Actual_mb"]

    display_cols = ["Actual_mb", "Predicted_mb", "Error_mb"]
    results.to_csv("prediction_ram_analysis.csv", index=False)


if __name__ == "__main__":
    main()
