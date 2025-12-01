import re
import subprocess
import time
import platform
import getpass
import yaml
import json
from pathlib import Path

import psutil
import csv

from training_manager.training_pipeline.io_utils.paths import Paths
from training_manager.training_pipeline.monitoring.hardware_monitor import HardwareMonitorContext

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

        run_base = RESULTS_DIR / MACHINE_NAME
        run_folder = run_base / run_id
        run_folder.mkdir(parents=True, exist_ok=True)
        log_dir = run_folder / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "stream.log"
        run_log_file = log_dir / "run_log.csv"
        # keep a copy of the config alongside the run
        try:
            (run_folder / "configuration.yaml").write_text(yaml_path.read_text())
        except Exception:
            pass

        cmd = [
            "mlagents-learn",
            str(yaml_path),
            f"--run-id={run_id}",
            f"--env={self.env_path}",
            "--no-graphics",
            f"--results-dir={run_base}",
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
            run_name=str(Path(MACHINE_NAME) / run_id),
            steps_per_log=10000,
            testing=False,
            results_dir=RESULTS_DIR,
        )
        hw_monitor = hw_monitor_ctx.monitor

        hw_monitor.record_initial_state()
        # Persist static hardware snapshot for rebuild/backfill
        try:
            (run_folder / "hardware_init.json").write_text(
                json.dumps(hw_monitor.initial_hardware_info, indent=2)
            )
        except Exception:
            pass
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

        # Write main summary row
        cfg_meta = load_config(yaml_path)
        hw_init = hw_monitor.initial_hardware_info
        hw_final = hw_monitor.final_hardware_info

        final_mean = hw_final.get("final_mean_reward", "NA")
        final_std = hw_final.get("final_std_reward", "NA")
        avg_cpu = hw_final.get("average_cpu_usage_percent", "NA")
        avg_ram = hw_final.get("average_ram_mb", "NA")
        peak_cpu = hw_final.get("peak_cpu_usage_percent", "NA")
        peak_ram = hw_final.get("peak_ram_mb", "NA")

        row = {k: "NA" for k in MAIN_HEADERS}
        row.update({
            "run_id": run_id,
            "machine_id": MACHINE_NAME,
            "run_log_file": str(run_log_file.relative_to(paths.root_dir)),
            "train_duration_s": round(end - start, 2),
            "avg_cpu_usage": avg_cpu,
            "avg_ram_usage": avg_ram,
            "peak_cpu_usage": peak_cpu,
            "peak_ram_usage": peak_ram,
            "final_mean_reward": final_mean,
            "final_std_reward": final_std,
            "time_to_convergence": "NA",
            "steps_to_convergence": "NA",
            "os_name": hw_init.get("operating_system", "NA"),
            "cpu_physical_cores": hw_init.get("cpu_physical_cores_count", "NA"),
            "cpu_logical_cores": hw_init.get("cpu_logical_cores_count", "NA"),
            "cpu_clock_ghz": hw_init.get("cpu_clock_speed_ghz", "NA"),
            "ram_mb": hw_init.get("total_ram_mb", "NA"),
        })
        row.update(cfg_meta)

        main_csv = RESULTS_DIR / "main.csv"
        append_main_row(main_csv, row)

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

# ---------------------------------------------------------------------------
# Helpers to parse config and write main summary CSV
# ---------------------------------------------------------------------------

# Schema for main summary CSV
MAIN_HEADERS = [
    "run_id", "machine_id", "run_log_file",
    "algo", "seed", "env_name",
    "os_name", "cpu_physical_cores", "cpu_logical_cores", "cpu_clock_ghz", "ram_mb",
    "avg_cpu_usage", "avg_ram_usage", "peak_cpu_usage", "peak_ram_usage",
    "learning_rate", "learning_rate_schedule", "batch_size", "buffer_size",
    "normalize", "hidden_units", "num_layers", "vis_encode_type", "gamma", "strength",
    "keep_checkpoints", "max_steps", "time_horizon", "summary_freq",
    "buffer_init_steps", "tau", "steps_per_update", "save_replay_buffer",
    "init_entcoef", "reward_signal_steps_per_update",
    "beta", "epsilon", "lambd", "num_epoch",
    "train_duration_s", "final_mean_reward", "final_std_reward",
    "time_to_convergence", "steps_to_convergence",
]


def load_config(yaml_path: Path) -> dict:
    """Parse YAML once; prepare defaults with NA for missing fields."""
    data = yaml.safe_load(yaml_path.read_text())
    behaviors = data.get("behaviors", {}) or {}
    env_name, env_cfg = next(iter(behaviors.items())) if behaviors else ("", {})
    trainer_type = env_cfg.get("trainer_type", "NA")

    hyper = env_cfg.get("hyperparameters", {}) or {}
    net = env_cfg.get("network_settings", {}) or {}
    reward = env_cfg.get("reward_signals", {}).get("extrinsic", {}) or {}

    def g(d, key):
        return d.get(key, "NA")

    return {
        "algo": trainer_type,
        "seed": env_cfg.get("seed", "NA"),
        "env_name": env_name,
        # shared
        "learning_rate": g(hyper, "learning_rate"),
        "learning_rate_schedule": g(hyper, "learning_rate_schedule"),
        "batch_size": g(hyper, "batch_size"),
        "buffer_size": g(hyper, "buffer_size"),
        "normalize": g(net, "normalize"),
        "hidden_units": g(net, "hidden_units"),
        "num_layers": g(net, "num_layers"),
        "vis_encode_type": g(net, "vis_encode_type"),
        "gamma": g(reward, "gamma"),
        "strength": g(reward, "strength"),
        "keep_checkpoints": env_cfg.get("keep_checkpoints", "NA"),
        "max_steps": env_cfg.get("max_steps", "NA"),
        "time_horizon": env_cfg.get("time_horizon", "NA"),
        "summary_freq": env_cfg.get("summary_freq", "NA"),
        # SAC-only
        "buffer_init_steps": g(hyper, "buffer_init_steps"),
        "tau": g(hyper, "tau"),
        "steps_per_update": g(hyper, "steps_per_update"),
        "save_replay_buffer": g(hyper, "save_replay_buffer"),
        "init_entcoef": g(hyper, "init_entcoef"),
        "reward_signal_steps_per_update": g(hyper, "reward_signal_steps_per_update") or g(hyper, "reward_signal_per_step"),
        # PPO-only
        "beta": g(hyper, "beta"),
        "epsilon": g(hyper, "epsilon"),
        "lambd": g(hyper, "lambd"),
        "num_epoch": g(hyper, "num_epoch"),
    }


def append_main_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAIN_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
