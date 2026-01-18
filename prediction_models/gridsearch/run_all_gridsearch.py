"""
Run grid search optimizations for all supported models and save results.
"""
import argparse
import csv
import json
from pathlib import Path

from GridSearchDecisionTree import GridSearchDecisionTree
from GridSearchRandomForest import GridSearchRandomForest
from GridSearchExtraTrees import GridSearchExtraTrees
from GridSearchGradientBoosting import GridSearchGradientBoosting
from GridSearchKnn import GridSearchKNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run grid search for all models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_all_gridsearch.py time_to_convergence\n"
            "  python run_all_gridsearch.py avg_ram_usage\n"
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        default="time_to_convergence",
        help="Target column name",
    )
    return p.parse_args()


def main():
    args = parse_args()
    target = args.target
    runners = [
        GridSearchDecisionTree,
        GridSearchRandomForest,
        GridSearchExtraTrees,
        GridSearchGradientBoosting,
        GridSearchKNN,
    ]
    results = []

    for cls in runners:
        print(f"\n=== Running {cls.__name__} ===")
        optimizer = cls(target=target)
        best_params = optimizer.run_optimization()
        results.append(
            {
                "model": cls.__name__,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

    out = Path(__file__).resolve().parent / f"gridsearch_results_{target}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "best_params"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved grid search results to: {out}")


if __name__ == "__main__":
    main()
