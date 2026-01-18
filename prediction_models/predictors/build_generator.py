import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        encoded_dataset = pd.read_csv(Path(__file__).resolve().parents[1] / f"encoded_datasets/encoded_{target}.csv")
        X = encoded_dataset.drop(columns=[target], errors="ignore")
        y = encoded_dataset[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        regr = LinearRegression()
        regr.fit(X_train, y_train)

        # save
        path = Path(__file__).resolve().parents[1] / f"predictors/builds/{target}_predictor.pkl"
        joblib.dump(regr, path)

        print(f"Linear regression model for {target} successfully built and saved to {path}")

    else:
        print("Provide target")
        print("Usage example: python build_generator.py avg_ram_usage")
        print("Usage example: python build_generator.py time_to_convergence")

if __name__ == "__main__":
    main()
