import argparse
import sys

from training_pipeline.training.batch_runner import BatchRunner
from training_pipeline.io_utils.paths import Paths
from training_pipeline.config_generator.generators.generator_ppo import run_ppo_generator
from training_pipeline.config_generator.generators.generator_sac import run_sac_generator
from training_pipeline.config_generator.generate_all import generate_all
from training_pipeline.cli.rebuild_main import rebuild

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML-Agents experiments and save results")

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run all YAML configs from configs directory using BatchRunner mode."
    )

    parser.add_argument(
        "--rebuild-main",
        action="store_true",
        help="Rebuild results/main.csv from existing run artifacts."
    )

    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--target-mean", type=float, default=95)
    parser.add_argument("--window-rows", type=int, default=5)
    parser.add_argument("--cv-max", type=float, default=0.10)
    parser.add_argument("--mean-jitter", type=float, default=2.0)
    parser.add_argument("--min-steps-before-check", type=int, default=200_000)

    return parser.parse_args()

def main():
    args = parse_args()
    paths = Paths()

    if args.rebuild_main and not args.batch:
        rebuild()
        return
    elif args.batch:
        print("\nRunning in batch mode...")
        runner = BatchRunner(
            rebuild_main=args.rebuild_main,
            max_steps=args.max_steps,
            target_mean=args.target_mean,
            window_rows=args.window_rows,
            cv_max=args.cv_max,
            mean_jitter=args.mean_jitter,
            min_steps_before_check=args.min_steps_before_check,
        )
        runner.run_forever()
        return
    else:
        print("Type 'mlrun -h' for help.")
        return

if __name__ == "__main__":
    main()
