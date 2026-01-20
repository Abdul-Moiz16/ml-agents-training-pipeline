from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from .build_accessor import get_hardware_specs, get_yaml_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest predictions on holdout runs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_holdout_backtest.py time_to_convergence\n"
            "  python run_holdout_backtest.py avg_ram_usage --save-individual\n"
            "  python run_holdout_backtest.py time_to_convergence --reuse-predictions\n"
        ),
    )
    p.add_argument("target", type=str, help="Target column name")
    p.add_argument(
        "--holdout-csv",
        type=str,
        default=None,
        help="Path to holdout_runs.csv",
    )
    p.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing holdout YAML configs",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path for backtest results",
    )
    p.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Use existing predicted_*.csv files if present",
    )
    p.add_argument(
        "--save-individual",
        action="store_true",
        help="Save predicted_<target>_<run_id>.csv files",
    )
    return p.parse_args()


def load_expected_cols(encoded_path: Path, target: str) -> List[str]:
    cols_ref = pd.read_csv(encoded_path, nrows=0)
    return [c for c in cols_ref.columns if c != target]


def predict_for_yaml(
    model,
    target: str,
    yaml_path: Path,
    expected_cols: List[str],
    run_id: str,
) -> float:
    yaml_params = get_yaml_params(yaml_path)
    hardware_specs = get_hardware_specs(run_id=run_id)
    input_data_dict = {**yaml_params, **hardware_specs}

    input_df = pd.DataFrame([input_data_dict])
    reordered_input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    pred = model.predict(reordered_input_df)[0]
    if target == "time_to_convergence":
        pred = np.expm1(pred)
    return float(pred)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    holdout_csv = (
        Path(args.holdout_csv)
        if args.holdout_csv
        else base_dir / "actual_and_predicted" / "holdout_runs.csv"
    )
    configs_dir = (
        Path(args.configs_dir)
        if args.configs_dir
        else base_dir / "configs_to_predict" / "holdout"
    )
    out_path = (
        Path(args.out)
        if args.out
        else base_dir / "actual_and_predicted" / f"backtest_results_{args.target}.csv"
    )

    if not holdout_csv.exists():
        raise FileNotFoundError(f"Holdout CSV not found: {holdout_csv}")
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    model_path = base_dir / "builds" / f"{args.target}_predictor.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    encoded_path = base_dir.parent / "encoded_datasets" / f"encoded_{args.target}.csv"
    if not encoded_path.exists():
        raise FileNotFoundError(f"Encoded dataset not found: {encoded_path}")

    expected_cols = load_expected_cols(encoded_path, args.target)
    holdout_df = pd.read_csv(holdout_csv)

    results = []
    actual_vals = []
    pred_vals = []

    for _, row in holdout_df.iterrows():
        run_id = str(row["run_id"])
        actual = row.get(args.target, "NA")
        try:
            actual_val = float(actual)
        except (TypeError, ValueError):
            continue

        pred_file = base_dir / "actual_and_predicted" / f"predicted_{args.target}_{run_id}.csv"
        if args.reuse_predictions and pred_file.exists():
            pred_df = pd.read_csv(pred_file)
            pred_val = float(pred_df[f"Predicted_{args.target}"].iloc[0])
        else:
            yaml_path = configs_dir / f"{run_id}.yaml"
            if not yaml_path.exists():
                continue
            model = joblib.load(model_path)
            pred_val = predict_for_yaml(model, args.target, yaml_path, expected_cols, run_id)

            if args.save_individual:
                pred_file.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {
                            "run_id": run_id,
                            f"Predicted_{args.target}": pred_val,
                        }
                    ]
                ).to_csv(pred_file, index=False)

        abs_err = abs(pred_val - actual_val)
        results.append(
            {
                "run_id": run_id,
                "machine_id": row.get("machine_id", "NA"),
                "algo": row.get("algo", "NA"),
                f"actual_{args.target}": actual_val,
                f"predicted_{args.target}": pred_val,
                "abs_error": abs_err,
            }
        )
        actual_vals.append(actual_val)
        pred_vals.append(pred_val)

    if not results:
        print("~! no backtest rows computed")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)

    mae = mean_absolute_error(actual_vals, pred_vals)
    med_ae = median_absolute_error(actual_vals, pred_vals)
    r2 = r2_score(actual_vals, pred_vals)

    summary_row = {
        "run_id": "SUMMARY",
        "machine_id": "ALL",
        "algo": "ALL",
        f"actual_{args.target}": "",
        f"predicted_{args.target}": "",
        "abs_error": "",
        "mae": mae,
        "median_ae": med_ae,
        "r2": r2,
    }

    results_df = pd.concat([pd.DataFrame([summary_row]), results_df], ignore_index=True)
    results_df.to_csv(out_path, index=False)

    print(f"Saved backtest results to: {out_path}")
    print(f"MAE: {mae:.3f}")
    print(f"Median AE: {med_ae:.3f}")
    print(f"R2: {r2:.3f}")


if __name__ == "__main__":
    main()
