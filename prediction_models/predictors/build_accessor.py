import argparse
import joblib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

def flatten_params(yaml_params):
    out = {}
    for key, val in yaml_params.items():
        if isinstance(val, dict):
            out.update(flatten_params(val))
        else:
            out[key] = val
    return out

def get_yaml_params(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_params = yaml.safe_load(f)

    flat_yaml_params = flatten_params(yaml_params)
    keys_to_remove = ['trainer_type', 'learning_rate_schedule','normalize', 'vis_encode_type', 'max_checkpoints']

    for key in keys_to_remove:
        flat_yaml_params.pop(key, None)

    return flat_yaml_params

def _normalize_os_name(name: str) -> str:
    return "macOS" if name == "Darwin" else name


def _local_hardware_specs() -> dict:
    os_name = _normalize_os_name(platform.system())
    cpu_physical_cores = psutil.cpu_count(logical=False)
    cpu_logical_cores = psutil.cpu_count(logical=True)

    try:
        cpu_freq = psutil.cpu_freq()
        cpu_clock_ghz = round(cpu_freq.current / 1000, 2) if cpu_freq and cpu_freq.current else None
    except (AttributeError, TypeError):
        cpu_clock_ghz = None

    memory_info = psutil.virtual_memory()
    ram_total_mb = memory_info.total / (1024 ** 2)

    return {
        "os_name": os_name,
        "cpu_physical_cores": cpu_physical_cores,
        "cpu_logical_cores": cpu_logical_cores,
        "cpu_clock_ghz": cpu_clock_ghz,
        "ram_mb": round(ram_total_mb, 2),
    }


def _specs_from_main_csv(run_id: str, main_csv: Path) -> dict | None:
    if not main_csv.exists():
        return None

    df = pd.read_csv(main_csv)
    if "run_id" not in df.columns:
        return None

    row = df.loc[df["run_id"] == run_id]
    if row.empty:
        return None

    row = row.iloc[0]
    def _num(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    os_name = row.get("os_name", None)
    os_name = _normalize_os_name(os_name) if isinstance(os_name, str) else None

    return {
        "os_name": os_name,
        "cpu_physical_cores": _num(row.get("cpu_physical_cores")),
        "cpu_logical_cores": _num(row.get("cpu_logical_cores")),
        "cpu_clock_ghz": _num(row.get("cpu_clock_ghz")),
        "ram_mb": _num(row.get("ram_mb")),
    }


def get_hardware_specs(run_id: str | None = None) -> dict:
    if run_id:
        main_csv = Path(__file__).resolve().parents[2] / "training_manager" / "experiments" / "results" / "main.csv"
        specs = _specs_from_main_csv(run_id, main_csv)
        if specs:
            return specs

    return _local_hardware_specs()

def _load_hardware_override(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Hardware override must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run predictor on a YAML config.")
    p.add_argument("target", type=str, help="Target column name")
    p.add_argument("yaml_path", type=str, help="Path to YAML config")
    p.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Path to YAML/JSON with hardware specs (os_name, cpu_*, ram_mb)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.target and args.yaml_path:
        target = args.target
        yaml_path = args.yaml_path

        model_path = Path(__file__).resolve().parents[1] / f"predictors/builds/{target}_predictor.pkl"
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        yaml_arg = Path(yaml_path)
        if yaml_arg.exists():
            yaml_params = get_yaml_params(yaml_arg)
            yaml_name = yaml_arg.stem
        else:
            yaml_params = get_yaml_params(
                Path(__file__).resolve().parent / "configs_to_predict" / yaml_path
            )
            yaml_name = Path(yaml_path).stem

        if args.hardware:
            hardware_specs = _load_hardware_override(Path(args.hardware))
        else:
            hardware_specs = get_hardware_specs(run_id=yaml_name)

        # Prefer explicit YAML values when present.
        input_data_dict = {**hardware_specs, **yaml_params}

        encoded_dataset_path = Path(__file__).resolve().parents[1] / f"encoded_datasets/encoded_{target}.csv"
        cols_ref = pd.read_csv(encoded_dataset_path, nrows=0)
        expected_cols = [c for c in cols_ref.columns if c != target]

        input_df = pd.DataFrame([input_data_dict])
        reordered_input_df = input_df.reindex(columns=expected_cols, fill_value=0)

        pred = model.predict(reordered_input_df)
        if target == "time_to_convergence":
            pred = np.expm1(pred)

        results = reordered_input_df.copy()
        results[f"Predicted_{target}"] = pred
        pred_path = (
            Path(__file__).resolve().parent
            / "actual_and_predicted"
            / f"predicted_{target}_{yaml_name}.csv"
        )
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(pred_path, index=False)

        print(f"Predicted: {pred}")
        print(f"Saved to: {pred_path}")

    else:
        print("Usage example: python build_accessor.py avg_ram_usage <config.yaml>")
        print("Usage example: python build_accessor.py time_to_convergence <config.yaml>")


if __name__ == "__main__":
    main()
