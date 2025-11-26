import argparse
import sys

from training_pipeline.training.batch_runner import BatchRunner
# from training_pipeline.training.single_runner import SingleRunner
from training_pipeline.io_utils.paths import Paths
from training_pipeline.config_generator.generators.generator_ppo import run_ppo_generator
from training_pipeline.config_generator.generators.generator_sac import run_sac_generator

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

    parser.add_argument(
        "--generate-configs",
        action="store_true",
        help="Generate training configuration files from with which to run bach training"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    paths = Paths()

    if args.generate_configs:
        print("\nGenerating PPO configs...")
        run_ppo_generator()
        print("\nPPO configs generated")

        print("\nGenerating SAC configs...")
        print("\nSAC configs generated")
        return
    elif args.batch:
        print("\nRunning in batch mode...")
        runner = BatchRunner()
        runner.run_all()
        return
    elif not args.config:
        print("\nConfiguration file is required for single run mode")
        return
    else:
        # TODO: Add single runner mode
        print("Single runner mode is under construction")
        return

if __name__ == "__main__":
    main()
