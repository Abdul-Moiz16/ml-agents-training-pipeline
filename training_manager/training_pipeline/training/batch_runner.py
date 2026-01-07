import shutil
import stat
import time
from pathlib import Path

from training_pipeline.training.single_runner import Runner, make_run_id, MACHINE_NAME
from training_pipeline.io_utils.paths import Paths
from training_pipeline.config_generator.generate_all import generate_all

from training_pipeline.cli.rebuild_main import rebuild

paths = Paths()
RESULTS_DIR = paths.results_dir
CONFIGS_DIR = paths.configs_dir

def generate_and_get_new_configs():
    """
    Calls generate_all() and returns ONLY the newly created YAML config paths.
    Works by comparing directory state before and after generation.
    """

    before = set(CONFIGS_DIR.glob("*.yaml"))
    generate_all()
    after = set(CONFIGS_DIR.glob("*.yaml"))

    # Newly created = set difference
    new_files = sorted(after - before)
    return new_files

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
    ):
        self.runner = Runner(
            max_steps=max_steps,
            target_mean=target_mean,
            window_rows=window_rows,
            cv_max=cv_max,
            mean_jitter=mean_jitter,
            min_steps_before_check=min_steps_before_check,
        )
        self.rebuild_main = rebuild_main

    def _maybe_rebuild(self):
        if self.rebuild_main:
            rebuild()

    @staticmethod
    def run_completed(run_id: str) -> str:
        run_base = RESULTS_DIR / MACHINE_NAME
        run_folder = run_base / run_id
        flag_file = run_folder / "complete.flag"
        # pick any behavior/output dir that is not run_logs
        if not run_folder.exists():
            return "fresh"

        behavior_dir = next(
            (d for d in run_folder.iterdir() if d.is_dir() and d.name != "run_logs"),
            None,
        )

        if flag_file.exists():
            return "done"

        if behavior_dir:
            if any(behavior_dir.glob("*.pt")) or any(behavior_dir.glob("*.onnx")):
                return "resume"

        return "stale"

    def run_all(self):
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))
        print(f"-----------------------------------------------------------------")
        print(f"~i found configs: {len(configs)}")
        print(f"-----------------------------------------------------------------")


        if not configs:
            print("~i no config files found")
            return

        for cfg in configs:
            run_id = make_run_id(cfg)
            status = self.run_completed(run_id)

            if status == "done":
                print(f"~i skip {run_id}: already completed")
                continue

            resume = status == "resume"

            if resume:
                print(f"~i resuming run: {run_id}")
            else:
                print(f"~i starting fresh run: {run_id}")

            self.runner.run(cfg, run_id, resume=resume)

            self._maybe_rebuild()

        print(f"-----------------------------------------------------------------")
        print("~i all runs completed")
        print(f"-----------------------------------------------------------------")

    def run_forever(self, sleep_seconds=3):
        """
        1. Run all existing configs exactly once.
        2. Then infinite loop:
            - Call generate_all()
            - Detect *only* the newly created configs
            - Run only those configs
            - Sleep
        """

        print("-----------------------------------------------------------------")
        print("~i running existing configs once")
        print("-----------------------------------------------------------------")

        # Run all existing configs ONCE
        self.run_all()

        print("-----------------------------------------------------------------")
        print("~i entering continuous config generation mode")
        print("-----------------------------------------------------------------")

        # Infinite generation → execution loop
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
            for cfg in new_cfgs:
                run_id = make_run_id(cfg)
                print(f"   → running {run_id}")
                self.runner.run(cfg, run_id, resume=False)

            print(f"~i sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)


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
