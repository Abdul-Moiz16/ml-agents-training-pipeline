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
            return self.builds_dir / "3DBall_Windows" / "UnityEnvironment.exe"
        elif system == "Linux":
            return self.builds_dir / "3DBall_Linux" / "3DBall.x86_64"
        elif system == "Darwin": # MacOS
            return self.builds_dir / "3DBall_MacOS" / "3DBall.app"
        else:
            raise RuntimeError(f"Unsopported OS: {system}")

        if not env_path.exists():
            raise FileNotFoundError(f"Unity environment not found: {env_path}")