from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

class DatasetEncoder:
    def __init__(self, target, data_path: Optional[Path] = None):
        self.data_path = data_path or Path(__file__).resolve().parents[1] / "training_manager" / "experiments" / "results" / "main.csv"
        self.target = target

        self.drop_cols: List[str] = [
            "run_id",
            "run_log_file",
            "train_duration_s",  # leakage
            "steps_to_convergence",  # leakage

            # Following features unavailable pre-run
            "avg_cpu_usage",
            "peak_ram_usage",
            "peak_cpu_usage",
            "final_std_reward",
            "final_mean_reward",

            # Following features highly influential for predictors (sort of cheating since if ran on another computer then predictor will be off)
            "machine_id",
            "os_name"
        ]

        if target == "time_to_convergence":
            self.drop_cols.append("avg_ram_usage")
        elif target == "avg_ram_usage":
            self.drop_cols.append("time_to_convergence")
        else:
            print("Unsupported target")

    def load_and_encode(self, target) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.data_path)

        df = df.drop(columns=self.drop_cols, errors="raise")

        df = df.replace("NA", pd.NA)
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna()

        # If target = time_to_convergence --> log transform the target since skewed
        if target == "time_to_convergence":
            df["time_to_convergence"] = np.log1p(df["time_to_convergence"])

        # One-hot encode remaining categorical/string columns so the regressor can consume them.
        X = df.drop(columns=[target], errors="ignore")

        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        return X, y


    def save_encoded(self, out_path: Optional[Path] = None) -> Path:
        """Persist the encoded feature matrix + target for inspection."""

        X, y = self.load_and_encode(self.target)

        encoded = X.copy()
        encoded[self.target] = y

        encoded.to_csv(out_path, index=False)
        print(f"Dataset saved to {out_path}")


def main():
    import sys

    if len(sys.argv) > 1:
        target_argv = sys.argv[1]
        try:
            encoder = DatasetEncoder(target=target_argv)
            output_filename = f"encoded_{target_argv}.csv"
            out_path = Path(__file__).resolve().parent / "encoded_datasets" / output_filename
            encoder.save_encoded(out_path=out_path)

            print(f"Encoded dataset saved to {out_path}")
        except Exception as e:
            print(f"Error: {e}")
            print("Target does not match any column in csv.")

    else:
        print("Provide target")
        print("Usage example: python dataset_encoder.py avg_ram_usage")
        print("Usage example: python dataset_encoder.py time_to_convergence")

if __name__ == "__main__":
    main()
