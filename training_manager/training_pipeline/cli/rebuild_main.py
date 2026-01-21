"""
Rebuild main.csv from existing run artifacts.

It scans training_manager/experiments/results/<run_id>/, looks for:
- configuration.yaml (hyperparameters/env/algo)
- run_log.csv (run log; takes last row for final CPU/RAM/time)

It writes/overwrites training_manager/experiments/results/main.csv with one row per run.
Hardware snapshot is recoverable from past runs by hardware_init.json.
"""

import csv
import json
from pathlib import Path
from typing import Dict

import yaml

from training_pipeline.training.single_runner import MAIN_HEADERS, load_config, MACHINE_NAME
from training_pipeline.io_utils.paths import Paths


def stats_from_run_log(path: Path) -> Dict[str, float]:
    cpu_vals = []
    ram_vals = []
    rows = [] # all rows
    conv_rows = [] # rows considered for convergence (skip baseline step 0)
    last = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            last = row
            try:
                cpu_vals.append(float(row.get("cpu_percent", "")))
            except (TypeError, ValueError):
                pass
            try:
                ram_vals.append(float(row.get("ram_mb", "")))
            except (TypeError, ValueError):
                pass
            # keep rows with a real step number for convergence
            try:
                step_val = int(row.get("step_number") or row.get("steps") or row.get("step") or 0)
            except (TypeError, ValueError):
                step_val = 0
            if step_val > 0:
                conv_rows.append(row)
    try:
        final_mean = float(last.get("mean_reward", ""))
    except (TypeError, ValueError):
        final_mean = "NA"
    try:
        final_std = float(last.get("std_of_reward", ""))
    except (TypeError, ValueError):
        final_std = "NA"

    # Convergence heuristic: require target mean over a small window with bounded noise
    steps_to_convergence = "NA"
    time_to_convergence = "NA"
    TARGET_MEAN = 100 # configurable: reward target to consider "converged"
    STD_RATIO = 1  # controls how much reward variability you allow relative to the mean. We compute std_of_reward / mean_reward for rows in the window; if that ratio is below STD_RATIO, the rewards are considered “stable enough.” A lower value means stricter stability (less noise allowed), a higher value means you’ll accept noisier rewards when deciding the run has converged.
    WINDOW_ROWS = 3 # require this many consecutive rows to meet target/noise

    if len(conv_rows) > 1:
        for i in range(WINDOW_ROWS - 1, len(conv_rows)):
            window = conv_rows[i - WINDOW_ROWS + 1 : i + 1]
            try:
                means = [float(r["mean_reward"]) for r in window]
                stds = [float(r["std_of_reward"]) for r in window]
                step_now = int(conv_rows[i].get("step_number") or conv_rows[i].get("steps") or conv_rows[i].get("step") or 0)
                time_now = float(conv_rows[i]["time_elapsed"])
            except (TypeError, ValueError, KeyError):
                continue

            if not means or any(m <= 0 for m in means):
                continue

            meets_target = all(m >= TARGET_MEAN for m in means)
            bounded_noise = all((s >= 0) and (m != 0) and (s / abs(m) <= STD_RATIO) for s, m in zip(stds, means))

            if meets_target and bounded_noise:
                steps_to_convergence = step_now
                time_to_convergence = time_now
                break

    return {
        "avg_cpu_usage": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else "NA",
        "avg_ram_usage": round(sum(ram_vals) / len(ram_vals), 2) if ram_vals else "NA",
        "peak_cpu_usage": round(max(cpu_vals), 2) if cpu_vals else "NA",
        "peak_ram_usage": round(max(ram_vals), 2) if ram_vals else "NA",
        "final_mean_reward": final_mean,
        "final_std_reward": final_std,
        "steps_to_convergence": steps_to_convergence,
        "time_to_convergence": time_to_convergence,
        "last": last,
    }


def rebuild():
    paths = Paths()
    results_root = paths.results_dir
    main_csv = results_root / "main.csv"
    rows = []

    # walk all run_log.csv under results/**/run_logs/ (supports machine_id/run_id structure) and read run data
    for run_log in results_root.rglob("run_logs/run_log.csv"):
        run_dir = run_log.parent.parent  # up from run_logs to run folder
        cfg_copy = run_dir / "configuration.yaml"
        hw_init_path = run_dir / "hardware_init.json"
        complete = run_dir / "complete.flag"

        if not (cfg_copy.exists() and run_log.exists() and complete.exists()):
            continue

        try:
            cfg_meta = load_config(cfg_copy)
        except Exception:
            cfg_meta = {}

        stats = stats_from_run_log(run_log)
        last = stats.get("last", {})

        try:
            hw_init = json.loads(hw_init_path.read_text()) if hw_init_path.exists() else {}
        except Exception:
            hw_init = {}

        machine_id = run_dir.parent.name if run_dir.parent != results_root else MACHINE_NAME
        row = {k: "NA" for k in MAIN_HEADERS}
        row.update(cfg_meta)
        row.update(
            {
                "run_id": run_dir.name,
                "machine_id": machine_id,
                "run_log_file": str(run_log.relative_to(paths.root_dir)),
                # hardware fields unavailable from past runs
                "os_name": hw_init.get("operating_system", "NA"),
                "cpu_physical_cores": hw_init.get("cpu_physical_cores_count", "NA"),
                "cpu_logical_cores": hw_init.get("cpu_logical_cores_count", "NA"),
                "cpu_clock_ghz": hw_init.get("cpu_clock_speed_ghz", "NA"),
                "ram_mb": hw_init.get("total_ram_mb", "NA"),
                "avg_cpu_usage": stats.get("avg_cpu_usage", "NA"),
                "avg_ram_usage": stats.get("avg_ram_usage", "NA"),
                "peak_cpu_usage": stats.get("peak_cpu_usage", "NA"),
                "peak_ram_usage": stats.get("peak_ram_usage", "NA"),
                "train_duration_s": last.get("time_elapsed", "NA"),
                "final_mean_reward": stats.get("final_mean_reward", "NA"),
                "final_std_reward": stats.get("final_std_reward", "NA"),
                "time_to_convergence": stats.get("time_to_convergence", "NA"),
                "steps_to_convergence": stats.get("steps_to_convergence", "NA"),
            }
        )
        rows.append(row)

    if not rows:
        print("No runs found to rebuild main.csv")
        return

    with main_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rebuilt main.csv with {len(rows)} runs -> {main_csv}")


if __name__ == "__main__":
    rebuild()
