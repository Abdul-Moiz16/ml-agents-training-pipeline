from pathlib import Path

from custom.scripts.config_generator.generators.generator_base import ConfigGeneratorBase
from custom.scripts.config_generator.parameters.params_sac import PARAMS_SAC

current_file = Path(__file__).resolve()

CONFIG_GENERATOR_DIR = current_file.parents[1]

CUSTOM_DIR = current_file.parents[3]

TEMPLATE_PATH = CONFIG_GENERATOR_DIR / "templates" / "sac_template.yaml"
OUTPUT_DIR = CUSTOM_DIR / "data" / "configs_for_training"

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
