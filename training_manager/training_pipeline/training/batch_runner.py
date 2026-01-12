import shutil
import stat
import time
import platform
import subprocess
from pathlib import Path

from training_pipeline.training.single_runner import Runner, make_run_id, MACHINE_NAME
from training_pipeline.io_utils.paths import Paths
from training_pipeline.config_generator.generate_all import generate_all

from training_pipeline.cli.rebuild_main import rebuild

paths = Paths()
RESULTS_DIR = paths.results_dir
CONFIGS_DIR = paths.configs_dir

# wait a bit between runs so ports get freed up
DELAY_BETWEEN_RUNS_S = 3


def cleanup_unity_processes(base_port: int | None = None):
    """
    Cleanup Unity processes/ports.

    If base_port is provided (parallel runs), avoid killing by process name to
    prevent terminating other workers; only clear the specific port.
    """
    system = platform.system()
    
    killed_any = False

    # Only kill by name when not running parallel (no base_port specified).
    if base_port is None:
        targets = ["3DBall", "UnityEnvironment"]
        for target in targets:
            try:
                if system == "Windows":
                    subprocess.run(
                        ["taskkill", "/F", "/IM", f"{target}*"],
                        capture_output=True,
                        timeout=10,
                    )
                else:
                    # mac/linux
                    result = subprocess.run(
                        ["pkill", "-9", "-f", target],
                        capture_output=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        killed_any = True
            except Exception:
                pass
    
    # also check if something is using the mlagents port
    port = base_port if base_port is not None else 5004
    try:
        if system == "Windows":
            # windows way - parse netstat output
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=10
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        try:
                            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)
                            killed_any = True
                        except Exception:
                            pass
        else:
            # mac/linux - lsof is way easier
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                        killed_any = True
                    except Exception:
                        pass
    except Exception:
        pass
    
    if killed_any:
        print("~i cleaned up lingering Unity processes")
    
    # give the OS a sec to actually release the ports
    time.sleep(3)
    
    return killed_any

def generate_and_get_new_configs():
    """
    generates configs and returns only the new ones
    (compares before/after to find what was added)
    """
    before = set(CONFIGS_DIR.glob("*.yaml"))
    generate_all()
    after = set(CONFIGS_DIR.glob("*.yaml"))

    new_files = sorted(after - before)
    return new_files


def cleanup_incomplete_runs():
    """
    deletes runs that got interrupted before hitting 500k steps
    these cant be resumed anyway (no checkpoint file) so we just nuke em
    and let them start fresh next time
    """
    import csv
    
    results_base = RESULTS_DIR / MACHINE_NAME
    if not results_base.exists():
        return
    
    deleted_count = 0
    
    for run_folder in results_base.iterdir():
        if not run_folder.is_dir():
            continue
        
        # Skip if already complete
        if (run_folder / "complete.flag").exists():
            continue
        # Skip if another worker is actively using this run
        if (run_folder / "inuse.flag").exists():
            continue
        
        # look for checkpoint files in the behavior folder (like 3DBall/)
        behavior_dir = next(
            (d for d in run_folder.iterdir() if d.is_dir() and d.name != "run_logs"),
            None,
        )
        
        has_checkpoint = False
        if behavior_dir:
            has_checkpoint = any(behavior_dir.glob("*.pt"))
        
        # if theres a checkpoint we can resume, dont touch it
        if has_checkpoint:
            continue
        
        # check if it actually did any training (steps > 0 in csv)
        run_log = run_folder / "run_logs" / "run_log.csv"
        last_step = 0
        if run_log.exists():
            try:
                with run_log.open('r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            step = int(row.get("step_number") or row.get("steps") or 0)
                            if step > last_step:
                                last_step = step
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
        
        # got some progress but no checkpoint = useless, delete it
        if last_step > 0:
            try:
                shutil.rmtree(run_folder)
                print(f"~i deleted incomplete run: {run_folder.name} (had {last_step} steps, no checkpoint)")
                deleted_count += 1
            except Exception as e:
                print(f"~! failed to delete {run_folder.name}: {e}")
    
    if deleted_count > 0:
        print(f"~i cleaned up {deleted_count} incomplete run(s)")


class BatchRunner:
    """Runs all training configs in the configs directory."""

    def __init__(
        self,
        rebuild_main: bool = False,
        max_steps: int = 10_000_000,
        target_mean: float = 95,
        window_rows: int = 5,
        cv_max: float = 0.10,
        mean_jitter: float = 2.0,
        min_steps_before_check: int = 200_000,
        base_port: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        inuse_timeout_hours: int = 6,
    ):
        self.runner = Runner(
            max_steps=max_steps,
            target_mean=target_mean,
            window_rows=window_rows,
            cv_max=cv_max,
            mean_jitter=mean_jitter,
            min_steps_before_check=min_steps_before_check,
            base_port=base_port,
        )
        self.rebuild_main = rebuild_main
        self.base_port = base_port
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.inuse_timeout_s = inuse_timeout_hours * 3600

    def _maybe_rebuild(self):
        if self.rebuild_main:
            rebuild()

    def run_completed(self, run_id: str) -> str:
        """
        checks whats up with a run:
        - done: has complete.flag, skip it
        - resume: has .pt checkpoint, we can continue from there
        - incomplete: has some progress but no checkpoint, cant resume
        - fresh: folder doesnt exist, start new
        - stale: folder exists but nothing useful in it
        """
        run_base = RESULTS_DIR / MACHINE_NAME
        run_folder = run_base / run_id
        flag_file = run_folder / "complete.flag"
        inuse_flag = run_folder / "inuse.flag"
        
        # No folder = fresh start
        if not run_folder.exists():
            return "fresh"

        # In-use flag check (avoid concurrent runs)
        if inuse_flag.exists():
            try:
                import json
                meta = json.loads(inuse_flag.read_text(encoding="utf-8"))
                started_ts = float(meta.get("started_ts", 0))
            except Exception:
                started_ts = 0

            if started_ts and (time.time() - started_ts) > self.inuse_timeout_s:
                try:
                    inuse_flag.unlink()
                    print(f"~i removed stale inuse.flag for {run_id}")
                except Exception:
                    return "inuse"
            else:
                return "inuse"

        # Complete flag exists = done, skip this run
        if flag_file.exists():
            return "done"

        # Check for checkpoint files (.pt)
        behavior_dir = next(
            (d for d in run_folder.iterdir() if d.is_dir() and d.name != "run_logs"),
            None,
        )
        
        has_checkpoint = False
        if behavior_dir:
            has_checkpoint = any(behavior_dir.glob("*.pt"))

        if has_checkpoint:
            return "resume"

        # check if theres any progress in the csv (catches runs that died before 500k)
        run_log = run_folder / "run_logs" / "run_log.csv"
        last_step = 0
        if run_log.exists():
            try:
                import csv
                with run_log.open('r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            step = int(row.get("step_number") or row.get("steps") or 0)
                            if step > last_step:
                                last_step = step
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass

        if last_step > 0:
            # got progress but no checkpoint = cant resume this
            return "incomplete"

        # Folder exists but no meaningful progress
        return "stale"

    def run_all(self):
        # Clean up incomplete runs (those with progress but no checkpoint)
        # so they can start fresh instead of being skipped
        cleanup_incomplete_runs()
        
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))
        print(f"-----------------------------------------------------------------")
        print(f"~i found configs: {len(configs)}")
        print(f"-----------------------------------------------------------------")

        if not configs:
            print("~i no config files found")
            return

        try:
            for i, cfg in enumerate(configs):
                if self.shard_count > 1 and (i % self.shard_count) != self.shard_index:
                    continue
                run_id = make_run_id(cfg)
                status = self.run_completed(run_id)

                if status == "done":
                    print(f"~i skip {run_id}: already completed")
                    continue
                if status == "inuse":
                    print(f"~i skip {run_id}: in use by another worker")
                    continue

                # kill any zombie unity processes before starting
                cleanup_unity_processes(self.base_port)

                resume = status == "resume"

                if resume:
                    print(f"~i resuming run: {run_id} (has checkpoint)")
                else:
                    print(f"~i starting fresh run: {run_id}")

                self.runner.run(cfg, run_id, resume=resume)

                self._maybe_rebuild()

                # wait a bit so ports get released
                if i < len(configs) - 1:
                    print(f"~i waiting {DELAY_BETWEEN_RUNS_S}s before next run...")
                    time.sleep(DELAY_BETWEEN_RUNS_S)

            print(f"-----------------------------------------------------------------")
            print("~i all runs completed")
            print(f"-----------------------------------------------------------------")
        except KeyboardInterrupt:
            print("\n-----------------------------------------------------------------")
            print("~i batch interrupted by user (Ctrl+C)")
            print("~i cleaning up lingering processes...")
            cleanup_unity_processes(self.base_port)
            print("~i exiting gracefully")
            print("-----------------------------------------------------------------")
            raise  # re-raise so the program actually stops

    def run_forever(self, sleep_seconds=3):
        """
        1. Run all existing configs exactly once.
        2. Then infinite loop:
            - Call generate_all()
            - Detect *only* the newly created configs
            - Run only those configs
            - Sleep
        """
        try:
            print("-----------------------------------------------------------------")
            print("~i running existing configs once")
            print("-----------------------------------------------------------------")

            self.run_all()

            print("-----------------------------------------------------------------")
            print("~i entering continuous config generation mode")
            print("-----------------------------------------------------------------")

            while True:
                print("\n~i generating new configs...")
                new_cfgs = generate_and_get_new_configs()

                if not new_cfgs:
                    print("~i no new configs generated (all duplicates?)")
                    time.sleep(sleep_seconds)
                    continue

                print(f"~i detected {len(new_cfgs)} new configs:")
                for cfg in new_cfgs:
                    print("   →", cfg.name)

                print("\n~i running new configs...")
                for i, cfg in enumerate(new_cfgs):
                    if self.shard_count > 1 and (i % self.shard_count) != self.shard_index:
                        continue
                    # Cleanup before each run
                    cleanup_unity_processes(self.base_port)
                    
                    run_id = make_run_id(cfg)
                    print(f"   → running {run_id}")
                    self.runner.run(cfg, run_id, resume=False)
                    
                    # Delay between runs
                    if i < len(new_cfgs) - 1:
                        print(f"~i waiting {DELAY_BETWEEN_RUNS_S}s before next run...")
                        time.sleep(DELAY_BETWEEN_RUNS_S)

                print(f"~i sleeping {sleep_seconds} seconds...")
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\n-----------------------------------------------------------------")
            print("~i batch interrupted by user (Ctrl+C)")
            print("~i cleaning up lingering processes...")
            cleanup_unity_processes(self.base_port)
            print("~i exiting gracefully")
            print("-----------------------------------------------------------------")


def force_delete(path: Path):
    """Recursively delete a path, handling read-only files."""
    path = Path(path)

    if not path.exists():
        return

    def remove_readonly(func, p, _):
        p = Path(p)
        if p.is_file():
            p.chmod(stat.S_IWRITE)
            func(p)
        elif p.is_dir():
            for sub in p.iterdir():
                force_delete(sub)
            p.chmod(stat.S_IWRITE)
            func(p)

    shutil.rmtree(path, onerror=remove_readonly)
