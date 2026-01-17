import joblib
import pandas as pd
import sys
from pathlib import Path
def main():
    if len(sys.argv) > 2:
        target = sys.argv[1]
        dataset_to_predict_path = sys.argv[2]

        model = joblib.load(Path(__file__).resolve().parents[1] / f"predictors/builds/{target}_predictor.pkl")
        test_data = pd.read_csv(sys.argv[2])

        preds = model.predict(dataset_to_predict_path)

        results = X_test.copy()
        results[f"Predicted_{target}"] = preds
        # TODO check if works correctly since untested (save to predicted_runs folder)
        results.to_csv(Path(__file__).resolve().parents[1] / f"predicted_runs/predicted_{dataset_to_predict_path}", index=False)

        print(preds)

    else:
        print("Usage example: python build_accessor.py avg_ram_usage as argv[1]")
        print("Usage example: python build_accessor.py time_to_convergence as argv[1]\n")
        print("Input path to dataset for prediction as an argv[2]")


if __name__ == "__main__":
    main()
