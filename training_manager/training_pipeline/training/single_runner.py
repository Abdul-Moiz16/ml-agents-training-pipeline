import re
import subprocess
import time
import platform
import getpass
from pathlib import Path

import psutil

from training_pipeline.io_utils.paths import Paths
from training_pipeline.monitoring.hardware_monitor import HardwareMonitorContext

paths = Paths()

RESULTS_DIR = paths.results_dir
UNITY_ENV_PATH = paths.unity_env


def _get_machine_name() -> str:
    """Build machine identifier from username and OS."""
    username = getpass.getuser()
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"
    return f"{username}_{os_name}"


MACHINE_NAME = _get_machine_name()

# Regex to extract training metrics from mlagents-learn output
# Example: "[INFO] 3DBall. Step: 10000. Time Elapsed: 20.983 s. Mean Reward: 1.175. Std of Reward: 0.734."
STAT_PATTERN = re.compile(
    r"Step:\s+(?P<steps>\d+)\.\s+Time\s+Elapsed:\s+(?P<time>[\d.]+)\s+s\.\s+"
    r"Mean\s+Reward:\s+(?P<mean>[-\d.]+)\.\s+Std\s+of\s+Reward:\s+(?P<std>[\d.]+)\.",
    re.IGNORECASE,
)


class Runner:
    """Runs a single ML-Agents training session."""

    def __init__(self, env_path: Path = UNITY_ENV_PATH):
        self.env_path = Path(env_path)

    def run(self, yaml_path: Path, run_id: str, resume: bool):
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"~ YAML file not found: {yaml_path}")
        if not self.env_path.exists():
            raise FileNotFoundError(f"~ Unity environment not found: {self.env_path}")

        run_folder = RESULTS_DIR / run_id
        run_folder.mkdir(parents=True, exist_ok=True)
        log_dir = run_folder / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "stream.log"

        cmd = [
            "mlagents-learn",
            str(yaml_path),
            f"--run-id={run_id}",
            f"--env={self.env_path}",
            "--no-graphics",
            f"--results-dir={RESULTS_DIR}",
        ]

        if resume:
            cmd.append("--resume")
        else:
            cmd.append("--force")

        print("-----------------------------------------------------------------")
        print("~i start training for:\n")
        print(f"Config: {yaml_path.name}")
        print(f"Run-ID: {run_id}")
        print(f"Env: {self.env_path}\n")
        print("~i training output:")
        print(cmd)

        start = time.time()

        hw_monitor_ctx = HardwareMonitorContext(
            run_name=run_id,
            steps_per_log=10000,
            testing=False,
            results_dir=RESULTS_DIR,
        )
        hw_monitor = hw_monitor_ctx.monitor

        hw_monitor.record_initial_state()
        hw_monitor.start_continuous_monitoring()

        try:
            with log_file.open("a", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                ps_proc = psutil.Process(proc.pid)
                ps_proc.cpu_percent(interval=None)  # prime the measurement
                hw_monitor._process = ps_proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    lf.write(line)
                    
                    m = STAT_PATTERN.search(line)
                    if m:
                        try:
                            step_num = int(m.group("steps"))
                            t_elapsed = float(m.group("time"))
                            mean_r = float(m.group("mean"))
                            std_r = float(m.group("std"))
                            print(
                                f"\n~hw CAPTURED step={step_num}, "
                                f"time={t_elapsed:.2f}s, "
                                f"mean_reward={mean_r:.3f}, "
                                f"std_reward={std_r:.3f}\n"
                            )
                        except (ValueError, KeyError):
                            continue

                        try:
                            hw_monitor.log_step(
                                step_number=step_num,
                                time_elapsed=t_elapsed,
                                mean_reward=mean_r,
                                std_of_reward=std_r,
                            )
                        except (ValueError, KeyError):
                            pass

                proc.wait()
                returncode = proc.returncode
        finally:
            hw_monitor.record_final_state()
            hw_monitor.save_to_csv()
        end = time.time()

        if returncode != 0:
            print("\n")
            print(f"~! training failed for: {yaml_path.name}")
            print(f"~! exit code: {returncode}")
            print("-----------------------------------------------------------------")
            return

        (run_folder / "complete.flag").write_text("ok")
        print("\n")
        print(f"~i training done for: {yaml_path.name}")
        print(f"~i duration: {round(end - start, 2)}s")
        print("-----------------------------------------------------------------")

        self.kill_unity_processes()

    @staticmethod
    def kill_unity_processes():
        """Kill hanging UnityEnvironment processes."""
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if p.info["name"] and "UnityEnvironment" in p.info["name"]:
                    print(f"~i Killing hanging Unity process PID={p.info['pid']}")
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def make_run_id(config_file: Path) -> str:
    return f"{MACHINE_NAME}_{config_file.stem}"
