import os
from pathlib import Path
from datetime import datetime
import shutil

import stat


from training_pipeline.training.single_runner import Runner, make_run_id
from training_pipeline.io_utils.paths import Paths

paths = Paths()
RESULTS_DIR = paths.results_dir
CONFIGS_DIR = paths.configs_dir


class BatchRunner:


    def __init__(self):
        self.runner = Runner()

    @staticmethod
    def run_completed(run_id: str) -> bool:

        run_folder = RESULTS_DIR / run_id
        flag_file = run_folder / "complete.flag"
        behavior_dir = next(run_folder.glob("*/"), None)

        # folder doesnt exist, start training
        if not run_folder.exists():
            return "fresh"

        # flag exists, dont start training again
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

