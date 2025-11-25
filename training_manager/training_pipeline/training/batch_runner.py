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

        # folder doesnt exist, start training
        if not run_folder.exists():
            return False

        # flag exists, dont start training again
        if flag_file.exists():
            return True

        # folder exists, flag doesnt exists
        print("-----------------------------------------------------------------")
        print(f"~i incomplete run detected: {run_id}")
        print(f"~i deleting old result folder: {run_folder}")

        force_delete(run_folder)

        print(f"~i result folder deleted: training will be restarted: {run_id}")
        print("-----------------------------------------------------------------")
        return False

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

            # check if run already exists
            if self.run_completed(run_id):
                print(f"-----------------------------------------------------------------")
                print(f"~i skip run {run_id}: already fully trained")
                print(f"-----------------------------------------------------------------")
                continue

            self.runner.run(cfg, run_id)

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

