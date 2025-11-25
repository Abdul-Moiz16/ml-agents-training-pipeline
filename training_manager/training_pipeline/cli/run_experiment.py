import argparse

from training_pipeline.training.batch_runner import BatchRunner
# from training_pipeline.training.single_runner import SingleRunner
from training_pipeline.io_utils.paths import Paths

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML-Agents experiments and save results")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a single .yaml config file for SingleRunner mode."
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run all YAML configs from configs directory using BatchRunner mode."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    paths = Paths()

    if args.batch:
        print("\nRunning in batch mode...")
        runner = BatchRunner()
        runner.run_all()
        return

    if not args.config:
        print("\nConfiguration file is required for single run mode")
        return

    # TODO: Add single runner mode
    print("Single runner mode is under construction")
    return

if __name__ == "__main__":
    main()
