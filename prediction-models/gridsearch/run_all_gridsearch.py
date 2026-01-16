"""
Run grid search optimizations for all supported models and save results.
"""
import csv
import json
from pathlib import Path

from GridSearchDecisionTree import GridSearchDecisionTree
from GridSearchRandomForest import GridSearchRandomForest
from GridSearchExtraTrees import GridSearchExtraTrees
from GridSearchGradientBoosting import GridSearchGradientBoosting
from GridSearchKnn import GridSearchKNN


def main():
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
        optimizer = cls()
        best_params = optimizer.run_optimization()
        results.append(
            {
                "model": cls.__name__,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

    out = Path(__file__).resolve().parent / "gridsearch_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "best_params"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved grid search results to: {out}")


if __name__ == "__main__":
    main()
