import argparse
import subprocess
import sys
from pathlib import Path

def _run_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.call(cmd)

def parse_args():
    p = argparse.ArgumentParser(prog="predrun", description="Run prediction_models pipeline steps")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("holdout", help="Create holdout runs + configs_to_predict/holdout")
    h.add_argument("--per-machine", type=int, default=5)

    e = sub.add_parser("encode", help="Encode dataset for a target (exclude holdout ids recommended)")
    e.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])
    e.add_argument("--exclude", type=str, default="prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt")

    fs = sub.add_parser("feature-select", help="Run feature selection (optional)")
    fs.add_argument("--target", required=True, choices=["time_to_convergence", "avg_ram_usage"])

    ma = sub.add_parser("model-analysis", help="Run model analysis")
    ma.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])
    ma.add_argument("--use-defaults", action="store_true")

    gs = sub.add_parser("gridsearch", help="Run gridsearch (optional)")
    gs.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])

    b = sub.add_parser("build", help="Build and save best predictor (.pkl)")
    b.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])

    bt = sub.add_parser("backtest", help="Run holdout backtest")
    bt.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])

    pr = sub.add_parser("predict", help="Predict for a single yaml config file")
    pr.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])
    pr.add_argument("yaml_file", type=str)

    prhw = sub.add_parser("predict-hw", help="Predict for folder of configs with hardware: block")
    prhw.add_argument("target", choices=["time_to_convergence", "avg_ram_usage"])
    prhw.add_argument("folder", type=str)

    allp = sub.add_parser("all", help="Run the typical full pipeline (holdout -> encode -> model-analysis -> build -> backtest)")
    allp.add_argument("--per-machine", type=int, default=5)
    allp.add_argument("--skip-feature-select", action="store_true")
    allp.add_argument("--skip-gridsearch", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()

    if args.cmd == "holdout":
        return _run_module("prediction_models.predictors.create_holdout", ["--per-machine", str(args.per_machine)])

    if args.cmd == "encode":
        return _run_module("prediction_models.dataset_encoder", [args.target, "--exclude", args.exclude])

    if args.cmd == "feature-select":
        return _run_module("prediction_models.feature_selection.feature_selection_runner", ["--target", args.target])

    if args.cmd == "model-analysis":
        mod_args = [args.target]
        if args.use_defaults:
            mod_args.append("--use-defaults")
        return _run_module("prediction_models.models_analysis.run_all_models", mod_args)

    if args.cmd == "gridsearch":
        return _run_module("prediction_models.gridsearch.run_all_gridsearch", [args.target])

    if args.cmd == "build":
        return _run_module("prediction_models.predictors.build_generator", [args.target])

    if args.cmd == "backtest":
        return _run_module("prediction_models.predictors.run_holdout_backtest", [args.target])

    if args.cmd == "predict":
        return _run_module("prediction_models.predictors.build_accessor", [args.target, args.yaml_file])

    if args.cmd == "predict-hw":
        return _run_module("prediction_models.predictors.build_accessor", [args.target, args.folder])

    if args.cmd == "all":
        rc = _run_module("prediction_models.predictors.create_holdout", ["--per-machine", str(args.per_machine)])
        if rc: return rc

        for tgt in ["time_to_convergence", "avg_ram_usage"]:
            rc = _run_module("prediction_models.dataset_encoder", [
                tgt, "--exclude", "prediction_models/predictors/actual_and_predicted/holdout_run_ids.txt"
            ])
            if rc: return rc

        if not args.skip_feature_select:
            for tgt in ["time_to_convergence", "avg_ram_usage"]:
                rc = _run_module("prediction_models.feature_selection.feature_selection_runner", ["--target", tgt])
                if rc: return rc

        if not args.skip_gridsearch:
            for tgt in ["time_to_convergence", "avg_ram_usage"]:
                rc = _run_module("prediction_models.gridsearch.run_all_gridsearch", [tgt])
                if rc: return rc

        for tgt in ["time_to_convergence", "avg_ram_usage"]:
            rc = _run_module("prediction_models.models_analysis.run_all_models", [tgt])
            if rc: return rc
            rc = _run_module("prediction_models.predictors.build_generator", [tgt])
            if rc: return rc
            rc = _run_module("prediction_models.predictors.run_holdout_backtest", [tgt])
            if rc: return rc

        return 0

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
