import shutil
import stat
from pathlib import Path

from training_manager.training_pipeline.training.single_runner import Runner, make_run_id, MACHINE_NAME
from training_manager.training_pipeline.io_utils.paths import Paths

paths = Paths()
RESULTS_DIR = paths.results_dir
CONFIGS_DIR = paths.configs_dir


class BatchRunner:
    """Runs all training configs in the configs directory."""

    def __init__(self):
        self.runner = Runner()

    @staticmethod
    def run_completed(run_id: str) -> str:
        run_base = RESULTS_DIR / MACHINE_NAME
        run_folder = run_base / run_id
        flag_file = run_folder / "complete.flag"
        # pick any behavior/output dir that is not run_logs
        behavior_dir = next(
            (d for d in run_folder.iterdir() if d.is_dir() and d.name != "run_logs"),
            None,
        )

        if not run_folder.exists():
            return "fresh"

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

        print(f"-----------------------------------------------------------------")
        print("~i all runs completed")
        print(f"-----------------------------------------------------------------")


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
