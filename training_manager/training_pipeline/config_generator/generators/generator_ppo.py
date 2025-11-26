from training_manager.training_pipeline.io_utils.paths import Paths

from training_manager.training_pipeline.config_generator.generators.generator_base import ConfigGeneratorBase
from training_manager.training_pipeline.config_generator.parameters.params_ppo import PARAMS_PPO

paths = Paths()

TEMPLATE_PATH = paths.config_generator_dir / "templates" / "ppo_template.yaml"
OUTPUT_DIR = paths.configs_dir

def run_ppo_generator():
    gen = ConfigGeneratorBase(
        template_path=TEMPLATE_PATH,
        param_dict=PARAMS_PPO,
        output_dir=OUTPUT_DIR,
        algorithm_name="ppo"
    )
    gen.generate()


if __name__ == "__main__":
    run_ppo_generator()
