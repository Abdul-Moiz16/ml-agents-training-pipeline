"""
Feature selection utilities for ML-Agents tabular dataset.

Methods included:
- SelectKBest with f_regression (fast linear filter)
- RFE with Ridge (wrapper, more stable than plain LinearRegression)
- Permutation importance with a tree ensemble (more reliable than impurity-based RF importances)

Examples:
    python feature_selection_runner.py --encoded encoded_dataset.csv --target time_to_convergence --k 15
    python feature_selection_runner.py --encoded encoded_dataset.csv --target avg_ram_usage --k 15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


def load_encoded_dataset(encoded_path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load encoded_dataset.csv and return numeric X, y."""
    
    df = pd.read_csv(encoded_path)

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found. Available columns (first 40): {list(df.columns[:40])} ..."
        )

    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target], errors="ignore")

    # Drop any helper columns if present
    for maybe_drop in ["converged"]:
        if maybe_drop in X.columns:
            X = X.drop(columns=[maybe_drop])

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    return X, y


def select_kbest_fregression(X: pd.DataFrame, y: pd.Series, k: int) -> pd.DataFrame:
    """Filter method: univariate linear relationship with y."""
    
    k = min(k, X.shape[1])
    
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    
    mask = selector.get_support()

    scores = pd.DataFrame({"feature": X.columns, "score": selector.scores_})
    selected = scores[mask].sort_values("score", ascending=False)
    
    return selected


def rfe_with_ridge(X: pd.DataFrame, y: pd.Series, k: int, alpha: float = 1.0) -> List[str]:
    """Wrapper method: RFE using Ridge for stability with correlated features."""
    
    k = min(k, X.shape[1])
    model = Ridge(alpha=alpha, random_state=42)
    
    selector = RFE(model, n_features_to_select=k, step=0.1)
    selector.fit(X, y)
    
    return list(X.columns[selector.get_support()])


def permutation_importance_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    
    model_name: str = "extratrees",
    test_size: float = 0.2,
    seed: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance on a held-out split; scored by negative MAE."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    elif model_name == "extratrees":
        model = ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    elif model_name == "gbr":
        model = GradientBoostingRegressor(random_state=seed)
    else:
        raise ValueError("model_name must be one of: rf, extratrees, gbr")

    model.fit(X_train, y_train)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class FeatureSelectionResults:
    kbest: pd.DataFrame
    rfe: List[str]
    perm: pd.DataFrame


def run_feature_selection(encoded_path: Path, target: str, k: int, perm_model: str) -> FeatureSelectionResults:
    X, y = load_encoded_dataset(encoded_path, target=target)

    # Drop rows where target is missing
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    kbest = select_kbest_fregression(X, y, k=k)
    rfe = rfe_with_ridge(X, y, k=k, alpha=1.0)
    perm = permutation_importance_ranking(X, y, model_name=perm_model)

    return FeatureSelectionResults(kbest=kbest, rfe=rfe, perm=perm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature selection on encoded ML-Agents dataset.")
    p.add_argument("--encoded", type=str, required=True, help="Path to encoded_dataset.csv")
    p.add_argument("--target", type=str, required=True, help="Target column name")
    p.add_argument("--k", type=int, default=15, help="Number of selected features (default 15)")
    p.add_argument(
        "--perm_model",
        type=str,
        default="extratrees",
        choices=["rf", "extratrees", "gbr"],
        help="Model used for permutation importance (default extratrees)",
    )
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    encoded_path = Path(args.encoded)
    if not encoded_path.exists():
        raise FileNotFoundError(f"Encoded dataset not found: {encoded_path}")

    res = run_feature_selection(encoded_path, target=args.target, k=args.k, perm_model=args.perm_model)

    out_dir = encoded_path.parent
    out_kbest = out_dir / f"feature_selection_kbest_{args.target}.csv"
    out_perm = out_dir / f"feature_selection_perm_{args.target}.csv"
    out_rfe = out_dir / f"feature_selection_rfe_{args.target}.txt"

    res.kbest.to_csv(out_kbest, index=False)
    res.perm.to_csv(out_perm, index=False)
    out_rfe.write_text("\n".join(res.rfe) + "\n", encoding="utf-8")

    print("\n=== SelectKBest (f_regression) top features linearly related to target ===")
    print(res.kbest.head(20).to_string(index=False))

    print("\n=== RFE (Ridge) selected features that works best together (linearly) ===")
    print(res.rfe)

    print("\n=== Permutation importance top features (what actually hurts prediction if we remove it) ===")
    print(res.perm.head(20).to_string(index=False))

    print("\nSaved:")
    print(" -", out_kbest)
    print(" -", out_rfe)
    print(" -", out_perm)


if __name__ == "__main__":
    main()
