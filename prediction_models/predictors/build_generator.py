import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_models.models_analysis.models import (
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    GradientBoostingModel,
    LinearRegressionModel,
    KNNModel,
)

def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        base_dir = Path(__file__).resolve().parents[1]
        encoded_path = base_dir / f"encoded_datasets/encoded_{target}.csv"
        comparisons_dir = base_dir / "models_analysis" / "models_comparisons"
        comparison_path = comparisons_dir / f"model_comparison_{target}_tuned.csv"
        if not comparison_path.exists():
            legacy_path = comparisons_dir / f"model_comparison_{target}.csv"
            if legacy_path.exists():
                comparison_path = legacy_path

        if not encoded_path.exists():
            raise FileNotFoundError(f"Encoded dataset not found: {encoded_path}")
        if not comparison_path.exists():
            raise FileNotFoundError(f"Model comparison not found: {comparison_path}")

        encoded_dataset = pd.read_csv(encoded_path)
        X = encoded_dataset.drop(columns=[target], errors="ignore")
        y = encoded_dataset[target]

        comparison = pd.read_csv(comparison_path)
        if "mean_ae" not in comparison.columns or "model" not in comparison.columns:
            raise ValueError("Model comparison CSV must include 'model' and 'mean_ae' columns.")

        comparison["mean_ae"] = pd.to_numeric(comparison["mean_ae"], errors="coerce")
        best_row = comparison.sort_values("mean_ae").iloc[0]
        best_model_name = str(best_row["model"])

        model_map = {
            "DecisionTreeModel": DecisionTreeModel,
            "RandomForestModel": RandomForestModel,
            "ExtraTreesModel": ExtraTreesModel,
            "GradientBoostingModel": GradientBoostingModel,
            "LinearRegressionModel": LinearRegressionModel,
            "KNNModel": KNNModel,
        }

        if best_model_name not in model_map:
            raise ValueError(f"Unknown model in comparison file: {best_model_name}")

        model_instance = model_map[best_model_name](target=target)
        model = model_instance.build_model()
        if not isinstance(model, Pipeline):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", model),
                ]
            )

        model.fit(X, y)

        # save
        path = base_dir / f"predictors/builds/{target}_predictor.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

        print(f"Selected model: {best_model_name} (mean_ae={best_row['mean_ae']})")
        print(f"Saved predictor to {path}")

    else:
        print("Provide target")
        print("Usage example: python build_generator.py avg_ram_usage")
        print("Usage example: python build_generator.py time_to_convergence")

if __name__ == "__main__":
    main()
