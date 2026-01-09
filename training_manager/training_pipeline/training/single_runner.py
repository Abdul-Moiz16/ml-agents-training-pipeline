import os
import re
import signal
import subprocess
import time
import platform
import getpass
import yaml
import json
from collections import deque
from pathlib import Path

import psutil
import csv

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

IS_WINDOWS = (platform.system() == "Windows")

# Example:
# "[INFO] 3DBall. Step: 10000. Time Elapsed: 20.983 s. Mean Reward: 1.175. Std of Reward: 0.734."
STAT_PATTERN = re.compile(
    r"Step:\s+(?P<steps>\d+)\.\s+Time\s+Elapsed:\s+(?P<time>[\d.]+)\s+s\.\s+"
    r"Mean\s+Reward:\s+(?P<mean>[-\d.]+)\.\s+Std\s+of\s+Reward:\s+(?P<std>[\d.]+)\.",
    re.IGNORECASE,
)


def _patch_max_steps(src_yaml: Path, dst_yaml: Path, max_steps: int) -> None:
    """
    Read src YAML, force behaviors/*/max_steps = max_steps, write to dst_yaml.
    This guarantees the 10M cap even if generator forgot it.
    """
    data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
    behaviors = data.get("behaviors") or {}

    if isinstance(behaviors, dict) and behaviors:
        for _, env_cfg in behaviors.items():
            if isinstance(env_cfg, dict):
                env_cfg["max_steps"] = int(max_steps)
    else:
        data["max_steps"] = int(max_steps)

    dst_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def popen_in_own_group(cmd: list[str], **kwargs) -> subprocess.Popen:
    if IS_WINDOWS:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # enables killpg(proc.pid, ...)
    return subprocess.Popen(cmd, **kwargs)

def send_interrupt(proc: subprocess.Popen) -> None:
    if IS_WINDOWS:
        # Avoid CTRL_BREAK_EVENT to prevent Fortran runtime abort; use terminate instead
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGINT)

def send_terminate(proc: subprocess.Popen) -> None:
    if IS_WINDOWS:
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)

def send_kill(proc: subprocess.Popen) -> None:
    if IS_WINDOWS:
        proc.kill()
    else:
        os.killpg(proc.pid, signal.SIGKILL)

def kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for c in parent.children(recursive=True):
        try:
            c.kill()
        except Exception:
            pass
    try:
        parent.kill()
    except Exception:
        pass

class Runner:
    """Runs a single ML-Agents training session."""

    def __init__(
        self,
        env_path: Path = UNITY_ENV_PATH,
        # hard cap
        max_steps: int = 10_000_000,
        # convergence = mean >= target AND stable
        target_mean: float = 99.5,
        window_rows: int = 5,          # consecutive stats lines required
        cv_max: float = 0.10,          # std/mean <= 10%  (stability)
        mean_jitter: float = 2,      # max(mean)-min(mean) within window
        min_steps_before_check: int = 200_000,
        # stopping behavior
        graceful_timeout_s: int = 120,  # wait after SIGINT before escalating
    ):
        self.env_path = Path(env_path)

        self.max_steps = int(max_steps)
        self.target_mean = float(target_mean)
        self.window_rows = int(window_rows)
        self.cv_max = float(cv_max)
        self.mean_jitter = float(mean_jitter)
        self.min_steps_before_check = int(min_steps_before_check)
        self.graceful_timeout_s = int(graceful_timeout_s)

    def _converged(self, means: list[float], stds: list[float]) -> bool:
        if len(means) < self.window_rows:
            return False
        if any(m < self.target_mean for m in means):
            return False

        # stability via coefficient-of-variation
        for m, s in zip(means, stds):
            if m <= 0:
                return False
            if (abs(s) / abs(m)) > self.cv_max:
                return False

        # prevent "alternating a lot" in the mean itself
        if (max(means) - min(means)) > self.mean_jitter:
            return False

        return True

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

        # Write a patched config copy (enforces max_steps) and run THAT
        cfg_copy = run_folder / "configuration.yaml"
        try:
            _patch_max_steps(yaml_path, cfg_copy, self.max_steps)
        except Exception:
            # fallback: still keep a copy
            try:
                cfg_copy.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

        cmd = [
            "mlagents-learn",
            str(cfg_copy),
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
        
        # tell the monitor if were resuming so it can trim the csv properly
        hw_monitor._is_resume = resume

        hw_monitor.record_initial_state()
        try:
            (run_folder / "hardware_init.json").write_text(
                json.dumps(hw_monitor.initial_hardware_info, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        hw_monitor.start_continuous_monitoring()

        # convergence bookkeeping
        mean_win = deque(maxlen=self.window_rows)
        std_win = deque(maxlen=self.window_rows)
        time_to_convergence = "NA"
        steps_to_convergence = "NA"

        stopped_intentionally = False
        stop_reason = "NA"
        stop_signal_sent = False
        returncode = None
        last_step_num = 0

        try:
            with log_file.open("a", encoding="utf-8") as lf:
                proc = popen_in_own_group(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                def request_stop(reason: str):
                    nonlocal stopped_intentionally, stop_reason, stop_signal_sent
                    if stop_signal_sent:
                        return
                    stop_signal_sent = True
                    stopped_intentionally = True
                    stop_reason = reason
                    print(f"\n~i stopping current run ({reason}) via interrupt...\n")
                    send_interrupt(proc)

                try:
                    ps_proc = psutil.Process(proc.pid)
                    ps_proc.cpu_percent(interval=None)
                    hw_monitor._process = ps_proc
                except psutil.NoSuchProcess:
                    pass

                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    lf.write(line)

                    m = STAT_PATTERN.search(line)
                    if not m:
                        continue

                    try:
                        step_num = int(m.group("steps"))
                        t_elapsed = float(m.group("time"))
                        mean_r = float(m.group("mean"))
                        std_r = float(m.group("std"))
                    except (ValueError, KeyError):
                        continue

                    last_step_num = step_num

                    # log to run_log.csv via your monitor
                    try:
                        hw_monitor.log_step(
                            step_number=step_num,
                            time_elapsed=t_elapsed,
                            mean_reward=mean_r,
                            std_of_reward=std_r,
                        )
                    except Exception:
                        pass

                    # ---- STOP 1: hard cap ----
                    if (not stop_signal_sent) and (step_num >= self.max_steps):
                        request_stop("max_steps")
                        # Don't break - drain remaining output

                    # ---- STOP 2: convergence (mean>=target AND stable) ----
                    mean_win.append(mean_r)
                    std_win.append(std_r)

                    if (
                        (not stop_signal_sent)
                        and step_num >= self.min_steps_before_check
                        and len(mean_win) == self.window_rows
                    ):
                        means = list(mean_win)
                        stds = list(std_win)
                        if self._converged(means, stds):
                            steps_to_convergence = step_num
                            time_to_convergence = t_elapsed
                            request_stop("converged")
                            # Don't break - drain remaining output

                # After EOF (process closed stdout), wait for process to exit
                try:
                    proc.wait(timeout=self.graceful_timeout_s)
                except subprocess.TimeoutExpired:
                    print("~! process timeout; sending terminate...")
                    send_terminate(proc)

                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print("~!! terminate timeout; killing...")
                        send_kill(proc)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        if IS_WINDOWS:
                            kill_process_tree(proc.pid)

                returncode = proc.returncode

                # Detect natural completion by ML-Agents (it reached its own max_steps)
                if returncode == 0 and not stopped_intentionally:
                    stop_reason = "natural_completion"
                    stopped_intentionally = True  # Treat as intentional for success check

        finally:
            hw_monitor.record_final_state()
            hw_monitor.save_to_csv()  # Always save hardware data (run_log.csv)
            end = time.time()

        # Determine if run completed successfully FIRST
        rc = returncode if returncode is not None else -999
        
        # Exit codes that indicate successful/intentional termination:
        # 0: normal exit
        # 130, -2: SIGINT (Ctrl+C or our interrupt)
        # 143, -15: SIGTERM (our terminate)
        # 137, -9: SIGKILL (our kill - last resort but still "successful" if we triggered it)
        ok_exit_codes = {0, 130, -2, 143, -15, 137, -9}

        # Treat intentional stops (converged/max_steps) as success even if rc is non-zero
        if stopped_intentionally and stop_reason in {"converged", "max_steps", "natural_completion"}:
            successful = True
        else:
            successful = stopped_intentionally and rc in ok_exit_codes

        if not successful:
            print("\n")
            print(f"~! training incomplete/failed for: {yaml_path.name}")
            print(f"~! exit code: {rc}")
            print(f"~! stopped_intentionally: {stopped_intentionally}")
            print(f"~! last_step: {last_step_num}")
            
            # if it crashed at 0 steps, delete the folder
            # this way it can start fresh next time instead of being stuck
            if last_step_num == 0:
                try:
                    import shutil
                    shutil.rmtree(run_folder)
                    print(f"~! Deleted failed run folder (0 steps): {run_folder.name}")
                    print(f"~! Will start fresh on next batch run")
                except Exception as e:
                    print(f"~! Could not delete run folder: {e}")
            else:
                print(f"~! Hardware data saved to run_log.csv (run can be resumed if checkpoint exists)")
            
            print("-----------------------------------------------------------------")
            return  # Don't write to main.csv for incomplete runs

        # === ONLY REACHED IF RUN COMPLETED SUCCESSFULLY ===
        
        # Write completion markers
        (run_folder / "complete.flag").write_text("ok", encoding="utf-8")
        (run_folder / "stop_reason.txt").write_text(stop_reason, encoding="utf-8")

        # Now write to main.csv (only for completed runs)
        cfg_meta = load_config(cfg_copy)
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
            "time_to_convergence": time_to_convergence,
            "steps_to_convergence": steps_to_convergence,
            "os_name": hw_init.get("operating_system", "NA"),
            "cpu_physical_cores": hw_init.get("cpu_physical_cores_count", "NA"),
            "cpu_logical_cores": hw_init.get("cpu_logical_cores_count", "NA"),
            "cpu_clock_ghz": hw_init.get("cpu_clock_speed_ghz", "NA"),
            "ram_mb": hw_init.get("total_ram_mb", "NA"),
        })
        row.update(cfg_meta)

        main_csv = RESULTS_DIR / "main.csv"
        append_main_row(main_csv, row)

        print("\n")
        print(f"~i training COMPLETED for: {yaml_path.name}")
        print(f"~i stop_reason: {stop_reason}")
        print(f"~i exit_code: {rc}")
        print(f"~i last_step: {last_step_num}")
        print(f"~i duration: {round(end - start, 2)}s")
        print(f"~i >> Added to main.csv")
        print("-----------------------------------------------------------------")


def make_run_id(config_file: Path) -> str:
    return f"{MACHINE_NAME}_{config_file.stem}"


# ---------------------------------------------------------------------------
# Helpers to parse config and write main summary CSV
# ---------------------------------------------------------------------------

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
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
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
        "buffer_init_steps": g(hyper, "buffer_init_steps"),
        "tau": g(hyper, "tau"),
        "steps_per_update": g(hyper, "steps_per_update"),
        "save_replay_buffer": g(hyper, "save_replay_buffer"),
        "init_entcoef": g(hyper, "init_entcoef"),
        "reward_signal_steps_per_update": g(hyper, "reward_signal_steps_per_update") or g(hyper, "reward_signal_per_step"),
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
