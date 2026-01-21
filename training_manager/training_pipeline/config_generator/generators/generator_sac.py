from pathlib import Path

from training_pipeline.io_utils.paths import Paths

from training_pipeline.config_generator.generators.generator_base import ConfigGeneratorBase
from training_pipeline.config_generator.parameters.params_sac import PARAMS_SAC

paths = Paths()

CONFIG_GENERATOR_DIR = paths.config_generator_dir

CUSTOM_DIR = paths.root_dir

TEMPLATE_PATH = CONFIG_GENERATOR_DIR / "templates" / "sac_template.yaml"
OUTPUT_DIR = paths.configs_dir

def run_sac_generator():
    gen = ConfigGeneratorBase(
        template_path=TEMPLATE_PATH,
        param_dict=PARAMS_SAC,
        output_dir=OUTPUT_DIR,
        algorithm_name="sac"
    )
    gen.generate()

if __name__ == "__main__":
    run_sac_generator()
