import platform
from pathlib import Path

class Paths:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.builds_dir = self.root_dir / "builds"
        self.experiments_dir = self.root_dir / "experiments"
        self.configs_dir = self.experiments_dir / "configs_for_training"
        self.results_dir = self.experiments_dir / "results"
        self.training_pipeline_dir = self.root_dir / "training_pipeline"
        self.config_generator_dir = self.training_pipeline_dir / "config_generator"
    
    @property
    def unity_env(self) -> Path:
        system = platform.system()

        if system == "Windows":
            env_path = self.builds_dir / "3DBall_Windows" / "UnityEnvironment.exe"
        elif system == "Linux":
            env_path = self.builds_dir / "3DBall_Linux" / "3DBall.x86_64"
        elif system == "Darwin":
            env_path = self.builds_dir / "3DBall_MacOS" / "3DBall.app"
        else:
            raise RuntimeError(f"Unsupported OS: {system}")
        
        return env_path