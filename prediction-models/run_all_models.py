"""
Train/test evaluation for multiple models using the shared preprocessing in BaseModel.
"""
from pathlib import Path
from BaselineModel import BaseModel

import pandas as pd

from models import (
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    GradientBoostingModel,
    LinearRegressionModel,
    KNNModel,
)

def main():
    results = []
    for cls in [
        DecisionTreeModel,
        RandomForestModel,
        ExtraTreesModel,
        GradientBoostingModel,
        LinearRegressionModel,
        KNNModel,
    ]:
        model_instance = cls()

        cv = model_instance.cross_validate_with_scaling(n_splits=5)

        results.append({
            "model": cls.__name__,
            "mean_ae": round(cv["mean_ae"], 3),
            "r2": round(cv["mean_r2"], 3),
            "median_ae": round(cv["median_ae"], 3),
        })

    df = pd.DataFrame(results).sort_values("mean_ae")
    out = Path(__file__).resolve().parent / "model_comparison_results.csv"
    df.to_csv(out, index=False)
    out = BaseModel().save_encoded()
    print("Saved:", out)
    print(df)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
