import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wrapper script to run ML-Agents training and log run data."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["PPO", "SAC"],
        help="RL algorithm to use (PPO or SAC).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the ML-Agents YAML config file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to use (will be passed to mlagents-learn and used in CSVs).",
    )
    return parser.parse_args()


def run_training_with_logging(algo: str, config_path: Path, run_id: str):
    """
    High-level orchestration function.

    For now: just a placeholder. Later it will:
      - read hyperparameters from YAML
      - start hardware monitoring
      - start mlagents-learn as a subprocess
      - watch logs and write to CSV
      - write summary to main CSV
    """
    print(f"[DEBUG] Would start training now: algo={algo}, config={config_path}, run_id={run_id}")
    # TODO: implement in later steps


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_training_with_logging(args.algo, config_path, args.run_id)


if __name__ == "__main__":
    main()
