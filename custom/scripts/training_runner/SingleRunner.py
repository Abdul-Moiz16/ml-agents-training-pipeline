import csv
import re
import subprocess
import time
import platform
from pathlib import Path

import psutil
import socket
import yaml

# --- ABSOLUTE FIXED PROJECT ROOT ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CUSTOM_DIR = PROJECT_ROOT / "custom"
CONFIG_DIR = CUSTOM_DIR / "data" / "configs_for_training"
RESULTS_DIR = CUSTOM_DIR / "data" / "results"
UNITY_ENV_PATH = CUSTOM_DIR / "builds" / "3DBall" / "UnityEnvironment.exe"

MACHINE_NAME = socket.gethostname()

STAT_PATTERN = re.compile(
    r"Step:\s*(?P<steps>\d+).*Time Elapsed:\s*(?P<time>[0-9.]+)\s*s.*Mean Reward:\s*(?P<mean>[-0-9.]+).*Std of Reward:\s*(?P<std>[-0-9.]+)",
    re.IGNORECASE,
)

HEADERS = [
    "run_id",
    "algo",
    "seed",
    "env_name",
    "run_log_file",
    "learning_rate",
    "learning_rate_schedule",
    "batch_size",
    "buffer_size",
    "normalize",
    "hidden_units",
    "num_layers",
    "vis_encode_type",
    "gamma",
    "strength",
    "keep_checkpoints",
    "max_steps",
    "time_horizon",
    "summary_freq",
    "buffer_init_steps",
    "tau",
    "steps_per_update",
    "save_replay_buffer",
    "init_entcoef",
    "reward_signal_steps_per_update",
    "beta",
    "epsilon",
    "lambd",
    "num_epoch",
    "os_name",
    "cpu_physical_cores",
    "cpu_logical_cores",
    "cpu_clock_ghz",
    "ram_mb",
    "avg_cpu_usage_proc",
    "avg_ram_usage_proc_mb",
    "avg_ram_usage_proc_pct",
    "avg_gpu_usage_proc",
    "steps",
    "time_elapsed",
    "mean_reward",
    "std_of_reward",
]


def snapshot_hardware() -> dict:
    """Capture static hardware info once per run."""
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    cpu_ghz = round(freq.current / 1000, 2) if freq else "NA"
    ram_mb = round(psutil.virtual_memory().total / (1024 ** 2), 2)
    return {
        "os_name": os_name,
        "cpu_physical_cores": cpu_phys,
        "cpu_logical_cores": cpu_log,
        "cpu_clock_ghz": cpu_ghz,
        "ram_mb": ram_mb,
    }


def read_gpu_usage_proc(_proc: psutil.Process):
    """Placeholder GPU util reader; return NA unless wired to pynvml/GPUtil."""
    return "NA"


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
        log_dir = run_folder / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "stream.log"
        progress_csv = run_folder / "progress.csv"

        cfg_meta = self.load_config(yaml_path)
        hw_meta = snapshot_hardware()

        cpu_samples = []
        mem_mb_samples = []
        mem_pct_samples = []
        gpu_samples = []

        cmd = [
            "mlagents-learn",
            str(yaml_path),
            f"--run-id={run_id}",
            f"--env={self.env_path}",
            "--no-graphics",
            f"--results-dir={RESULTS_DIR}",
            "--force",
        ]

        print("-----------------------------------------------------------------")
        print("~i start training for:\n")
        print(f"Config: {yaml_path.name}")
        print(f"Run-ID: {run_id}")
        print(f"Env: {self.env_path}\n")
        print("~i training output:")
        print(cmd)

        start = time.time()
        with log_file.open("a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            ps_proc = psutil.Process(proc.pid)
            # Prime CPU percent measurement for the process
            ps_proc.cpu_percent(interval=None)

            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
                m = STAT_PATTERN.search(line)
                if m:
                    # sample process-only usage when a progress line appears
                    cpu_samples.append(ps_proc.cpu_percent(interval=None))
                    mem_info = ps_proc.memory_info()
                    mem_mb_samples.append(mem_info.rss / (1024 ** 2))
                    mem_pct_samples.append(ps_proc.memory_percent())
                    gpu_samples.append(read_gpu_usage_proc(ps_proc))

                    row = self.build_row(
                        cfg_meta,
                        run_id,
                        str(log_file),
                        m,
                        hw_meta,
                        cpu_samples,
                        mem_mb_samples,
                        mem_pct_samples,
                        gpu_samples,
                    )
                    self.append_csv(progress_csv, row)
            proc.wait()
            returncode = proc.returncode
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
    def append_csv(path: Path, row: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def load_config(self, yaml_path: Path) -> dict:
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
            "reward_signal_steps_per_update": g(
                hyper, "reward_signal_steps_per_update"
            )
            or g(hyper, "reward_signal_per_step"),
            # PPO-only
            "beta": g(hyper, "beta"),
            "epsilon": g(hyper, "epsilon"),
            "lambd": g(hyper, "lambd"),
            "num_epoch": g(hyper, "num_epoch"),
        }

    def build_row(
        self,
        cfg: dict,
        run_id: str,
        log_file: str,
        match: re.Match,
        hw_meta: dict,
        cpu_samples,
        mem_mb_samples,
        mem_pct_samples,
        gpu_samples, ) -> dict:
        
        row = {k: "NA" for k in HEADERS}
        row.update(cfg)
        row.update(hw_meta)

        avg_cpu = round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else "NA"
        avg_mem_mb = (
            round(sum(mem_mb_samples) / len(mem_mb_samples), 2)
            if mem_mb_samples
            else "NA"
        )
        avg_mem_pct = (
            round(sum(mem_pct_samples) / len(mem_pct_samples), 2)
            if mem_pct_samples
            else "NA"
        )
        numeric_gpu = [g for g in gpu_samples if isinstance(g, (int, float))]
        avg_gpu = round(sum(numeric_gpu) / len(numeric_gpu), 2) if numeric_gpu else "NA"

        row.update(
            {
                "run_id": run_id,
                "run_log_file": log_file,
                "avg_cpu_usage_proc": avg_cpu,
                "avg_ram_usage_proc_mb": avg_mem_mb,
                "avg_ram_usage_proc_pct": avg_mem_pct,
                "avg_gpu_usage_proc": avg_gpu,
                "steps": match.group("steps"),
                "time_elapsed": match.group("time"),
                "mean_reward": match.group("mean"),
                "std_of_reward": match.group("std"),
            }
        )
        return row

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
    return f"{MACHINE_NAME}_{config_file.stem}"
