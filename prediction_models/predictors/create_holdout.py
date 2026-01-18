from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def resolve_config_path(results_root: Path, machine_id: str, run_id: str) -> Optional[Path]:
    candidate = results_root / machine_id / run_id / "configuration.yaml"
    if candidate.exists():
        return candidate
    legacy = results_root / run_id / "configuration.yaml"
    if legacy.exists():
        return legacy
    return None


def select_holdout(
    df: pd.DataFrame,
    results_root: Path,
    per_machine: int,
    seed: int,
    require_complete: bool,
    require_converged: bool,
) -> List[dict]:
    holdout = []
    for machine_id, group in df.groupby("machine_id", dropna=True):
        group = group.sample(frac=1, random_state=seed)
        picked = 0
        for _, row in group.iterrows():
            run_id = str(row["run_id"])
            if require_converged:
                try:
                    ttc = float(row.get("time_to_convergence", "nan"))
                except (TypeError, ValueError):
                    ttc = float("nan")
                if np.isnan(ttc):
                    continue

            cfg_path = resolve_config_path(results_root, machine_id, run_id)
            if cfg_path is None:
                continue
            if require_complete and not (cfg_path.parent / "complete.flag").exists():
                continue

            holdout.append(
                {
                    "run_id": run_id,
                    "machine_id": machine_id,
                    "algo": row.get("algo", "NA"),
                    "time_to_convergence": row.get("time_to_convergence", "NA"),
                    "avg_ram_usage": row.get("avg_ram_usage", "NA"),
                    "config_path": str(cfg_path),
                }
            )
            picked += 1
            if picked >= per_machine:
                break

        if picked < per_machine:
            print(f"~! only picked {picked}/{per_machine} for {machine_id}")

    return holdout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create holdout runs for backtesting.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python create_holdout.py\n"
            "  python create_holdout.py --per-machine 2 --seed 123\n"
        ),
    )
    p.add_argument("--per-machine", type=int, default=1, help="Holdout runs per machine_id")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / "training_manager" / "experiments" / "results"
    main_csv = results_root / "main.csv"

    if not main_csv.exists():
        raise FileNotFoundError(f"main.csv not found: {main_csv}")

    df = pd.read_csv(main_csv)
    df = df.dropna(subset=["run_id", "machine_id"])

    holdout_rows = select_holdout(
        df,
        results_root=results_root,
        per_machine=args.per_machine,
        seed=args.seed,
        require_complete=True,
        require_converged=True,
    )

    if not holdout_rows:
        print("~! no holdout runs selected")
        return

    holdout_df = pd.DataFrame(holdout_rows)

    predictors_dir = Path(__file__).resolve().parent
    configs_dir = predictors_dir / "configs_to_predict" / "holdout"
    actual_dir = predictors_dir / "actual_and_predicted"
    configs_dir.mkdir(parents=True, exist_ok=True)
    actual_dir.mkdir(parents=True, exist_ok=True)

    holdout_csv = actual_dir / "holdout_runs.csv"
    holdout_df.to_csv(holdout_csv, index=False)

    holdout_ids_path = actual_dir / "holdout_run_ids.txt"
    holdout_ids_path.write_text(
        "\n".join(holdout_df["run_id"].astype(str).tolist()) + "\n",
        encoding="utf-8",
    )

    copied = 0
    for _, row in holdout_df.iterrows():
        src = Path(row["config_path"])
        dst = configs_dir / f"{row['run_id']}.yaml"
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            copied += 1

    print(f"~i selected {len(holdout_df)} holdout runs")
    print(f"~i wrote {holdout_csv}")
    print(f"~i wrote {holdout_ids_path}")
    print(f"~i copied {copied} configs to {configs_dir}")


if __name__ == "__main__":
    main()
