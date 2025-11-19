import subprocess
from pathlib import Path
from datetime import datetime

import psutil
import os

from pathlib import Path

# --- ABSOLUTE FIXED PROJECT ROOT ---
PROJECT_ROOT = Path(r"C:\Users\nico\PycharmProjects\Group8-AI-ML") # please change before use

CUSTOM_DIR = PROJECT_ROOT / "custom"

CONFIG_DIR = CUSTOM_DIR / "data" / "configs_for_training"

RESULTS_DIR = CUSTOM_DIR / "data" / "results"

UNITY_ENV_PATH = CUSTOM_DIR / "builds" / "3DBall" / "UnityEnvironment.exe"

INITIALS = "NM" # please change before use

class Runner:
    """starts a single ml agent training"""

    def __init__(self, env_path: Path = UNITY_ENV_PATH):
        self.env_path = Path(env_path)

    def run(self, yaml_path: Path, run_id: str):
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"~ YAML file not found: {yaml_path}")

        if not self.env_path.exists():
            raise FileNotFoundError(f"~ Unity environment not found: {self.env_path}")

        run_folder = RESULTS_DIR / run_id
        run_folder.mkdir(parents=True, exist_ok=True)

        # ML-Agents CLI command
        cmd = [
            "mlagents-learn",
            str(yaml_path),
            f"--run-id={run_id}",
            f"--env={self.env_path}",
            "--no-graphics",
            f"--results-dir={RESULTS_DIR}",
            "--force",
        ]

        # Logging
        print(f"-----------------------------------------------------------------")
        print(f"~i start training for:")
        print("\n")
        print(f"Config: {yaml_path.name}")
        print(f"Run-ID: {run_id}")
        print(f"Env: {self.env_path}")
        print("\n")

        print("~i training output:")
        print(cmd)
        # start training
        result = subprocess.run(cmd, shell=True)

        run_folder = RESULTS_DIR / run_id

        if result.returncode != 0:
            print("\n")
            print(f"~! training failed for: {yaml_path.name}")
            print(result.stderr.decode(errors="ignore"))
            print("-----------------------------------------------------------------")
            return

        else:
            # success
            (run_folder / "complete.flag").write_text("ok")
            print("\n")
            print(f"~i training done for: {yaml_path.name}\n")
            print(f"-----------------------------------------------------------------")

        self.kill_unity_processes()

    @staticmethod
    def kill_unity_processes():
        """Kill all UnityEnvironment processes to avoid WorkerInUseException."""
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if p.info["name"] and "UnityEnvironment" in p.info["name"]:
                    print(f"~i Killing hanging Unity process PID={p.info['pid']}")
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def make_run_id(config_file: Path) -> str:

   # date = datetime.now().strftime("%Y-%m-%d")
    return f"{INITIALS}_{config_file.stem}"



