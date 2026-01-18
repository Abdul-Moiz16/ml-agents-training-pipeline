import joblib
import platform

import numpy as np
import psutil
import sys
from pathlib import Path
import yaml
import pandas as pd

# TODO add pc names to avoid confusion of output predictions (for which pc they were)

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

def get_hardware_specs():
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"

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
        "cpu_physical_cores": cpu_physical_cores,
        "cpu_logical_cores": cpu_logical_cores,
        "cpu_clock_ghz": cpu_clock_ghz,
        "ram_mb": round(ram_total_mb, 2)
    }

def main():
    if len(sys.argv) > 2:
        target = sys.argv[1]
        yaml_path = sys.argv[2]

        model_path = Path(__file__).resolve().parents[1] / f"predictors/builds/{target}_predictor.pkl"
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        yaml_params = get_yaml_params(yaml_path)
        hardware_specs = get_hardware_specs()

        input_data_dict = {**yaml_params, **hardware_specs}

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
        # TODO check if works correctly since untested (save to predicted_runs folder)
        yaml_name = Path(yaml_path).stem
        pred_path = Path(__file__).resolve().parents[1] / f"predicted_runs/predicted_{target}_{yaml_name}.csv"
        results.to_csv(pred_path, index=False)

        print(f"Predicted: {pred}")
        print(f"Saved to: {pred_path}")

    else:
        print("Usage example: python build_accessor.py avg_ram_usage as argv[1]")
        print("Usage example: python build_accessor.py time_to_convergence as argv[1]\n")
        print("Input path to yaml config for prediction as an argv[2]")


if __name__ == "__main__":
    main()
