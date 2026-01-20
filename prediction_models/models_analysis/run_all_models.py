"""
Train/test evaluation for multiple models using the shared preprocessing in BaseModel.
"""

import pandas as pd
from models import (
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    GradientBoostingModel,
    LinearRegressionModel,
    KNNModel,
)
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from dataset_encoder import DatasetEncoder

def main():
    import sys
    # Get encoded dataset specific to target
    # If it does not exist yet, generate it
    if len(sys.argv) > 1:
        use_defaults = "--use-defaults" in sys.argv
        target_argv = next(
            (arg for arg in sys.argv[1:] if not arg.startswith("-")),
            None,
        )
        if not target_argv:
            print("Provide target")
            print("Usage example: python run_all_models.py time_to_convergence")
            print("Usage example: python run_all_models.py avg_ram_usage --use-defaults")
            return
        try:
            filename = f"encoded_{target_argv}.csv"
            encoded_dataset_path = Path(__file__).resolve().parents[1] / "encoded_datasets" / filename

            if encoded_dataset_path.exists():
                print("Found encoded dataset")
            else:
                print(f"Encoded dataset not found. Generating {filename}")
                dataset_encoder = DatasetEncoder(target=target_argv)
                dataset_encoder.save_encoded(out_path=encoded_dataset_path)

            encoded_dataset = pd.read_csv(encoded_dataset_path)
            X = encoded_dataset.drop(columns=[target_argv], errors="ignore")
            y = encoded_dataset[target_argv]

            # Run the models
            results = []
            for cls in [
                DecisionTreeModel,
                RandomForestModel,
                ExtraTreesModel,
                GradientBoostingModel,
                LinearRegressionModel,
                KNNModel,
            ]:
                model_instance = cls(target=target_argv, use_defaults=use_defaults)

                transform_log = (target_argv == "time_to_convergence")
                cv = model_instance.cross_validate_with_scaling(X, y, n_splits=5, transform_log=transform_log)

                results.append({
                    "model": cls.__name__,
                    "mean_ae": round(cv["mean_ae"], 3),
                    "r2": round(cv["mean_r2"], 3),
                    "median_ae": round(cv["median_ae"], 3),
                })

            df = pd.DataFrame(results).sort_values("mean_ae")
            df.insert(0, "mode", "default" if use_defaults else "tuned")
            suffix = "default" if use_defaults else "tuned"
            out = (
                Path(__file__).resolve().parent
                / f"models_comparisons/model_comparison_{target_argv}_{suffix}.csv"
            )
            df.to_csv(out, index=False)
            print("Saved model comparison to:", out)
            print(df)

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Provide target")
        print("Usage example: python run_all_models.py avg_ram_usage")
        print("Usage example: python run_all_models.py time_to_convergence")


if __name__ == "__main__":
    main()
