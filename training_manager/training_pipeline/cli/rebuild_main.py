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

from training_manager.training_pipeline.training.single_runner import MAIN_HEADERS, load_config, MACHINE_NAME
from training_manager.training_pipeline.io_utils.paths import Paths


def stats_from_run_log(path: Path) -> Dict[str, float]:
    cpu_vals = []
    ram_vals = []
    last = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
            try:
                cpu_vals.append(float(row.get("cpu_percent", "")))
            except (TypeError, ValueError):
                pass
            try:
                ram_vals.append(float(row.get("ram_mb", "")))
            except (TypeError, ValueError):
                pass
    return {
        "avg_cpu_usage": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else "NA",
        "avg_ram_usage": round(sum(ram_vals) / len(ram_vals), 2) if ram_vals else "NA",
        "peak_cpu_usage": round(max(cpu_vals), 2) if cpu_vals else "NA",
        "peak_ram_usage": round(max(ram_vals), 2) if ram_vals else "NA",
        "last": last,
    }


def rebuild():
    paths = Paths()
    results_root = paths.results_dir
    main_csv = results_root / "main.csv"
    rows = []

    # walk all run_log.csv under results/**/run_logs/ (supports machine_id/run_id structure)
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
                "final_ram_usage": last.get("ram_mb") or last.get("avg_ram_usage") or "NA",
                "final_cpu_usage": last.get("cpu_percent") or last.get("avg_cpu_usage") or "NA",
                "time_to_convergence": "NA",
                "steps_to_convergence": "NA",
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
