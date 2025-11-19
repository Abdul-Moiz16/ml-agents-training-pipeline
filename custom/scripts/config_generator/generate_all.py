from generators.generator_sac import run_sac_generator
from generators.generator_ppo import run_ppo_generator

if __name__ == "__main__":
    run_sac_generator()
    run_ppo_generator()
    print("\n~ all configs generated ")
