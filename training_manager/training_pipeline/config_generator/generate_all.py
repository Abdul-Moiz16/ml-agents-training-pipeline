from training_pipeline.config_generator.generators.generator_sac import run_sac_generator
from training_pipeline.config_generator.generators.generator_ppo import run_ppo_generator

def generate_all():
    run_sac_generator()
    run_ppo_generator()
    print("\n~ all configs generated ")
