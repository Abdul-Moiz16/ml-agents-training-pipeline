"""
Hyperparameter tuning for KNN model to find optimal number of neighbors (k).

Tests different k values and reports MAE/R² for each, identifying the best k.
"""
from pathlib import Path

import pandas as pd

from models import KNNModel


def main():
    # Test a range of k values
    k_values = [3, 5, 7, 9, 11, 15, 21, 25, 31]
    
    results = []
    for k in k_values:
        model = KNNModel(n_neighbors=k)
        mae, r2, _ = model.train_test_eval(test_size=0.2, seed=42)
        results.append({
            "k": k,
            "mae": round(mae, 3),
            "r2": round(r2, 3)
        })
        print(f"k={k:2d}  |  MAE={mae:8.3f}  |  R²={r2:.3f}")

    df = pd.DataFrame(results)
    
    # Find best k based on lowest MAE
    best_idx = df["mae"].idxmin()
    best_k = df.loc[best_idx, "k"]
    best_mae = df.loc[best_idx, "mae"]
    best_r2 = df.loc[best_idx, "r2"]
    
    print("\n" + "=" * 50)
    print(f"Best k: {best_k}")
    print(f"  MAE: {best_mae}")
    print(f"  R²:  {best_r2}")
    print("=" * 50)
    
    # Save results to CSV
    out = Path(__file__).resolve().parent / "knn_tuning_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
