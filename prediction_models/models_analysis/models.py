import json
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from .BaselineModel import BaseModel
except ImportError:
    from BaselineModel import BaseModel


def _load_gridsearch_params(target: str | None, model_key: str, prefix: str) -> dict:
    if not target:
        return {}

    results_path = (
        Path(__file__).resolve().parents[1]
        / "gridsearch"
        / f"gridsearch_results_{target}.csv"
    )
    if not results_path.exists():
        return {}

    try:
        df = pd.read_csv(results_path)
    except Exception:
        return {}

    row = df.loc[df["model"] == model_key]
    if row.empty:
        return {}

    try:
        raw_params = json.loads(row.iloc[0]["best_params"])
    except Exception:
        return {}

    params = {}
    for key, val in raw_params.items():
        if prefix and key.startswith(prefix):
            params[key[len(prefix):]] = val
        else:
            params[key] = val
    return params


class DecisionTreeModel(BaseModel):
    def __init__(self, data_path=None, target: str | None = None, use_defaults: bool = False):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)

    def build_model(self):
        if self.use_defaults:
            return DecisionTreeRegressor(random_state=42)

        params = _load_gridsearch_params(
            self.target, "GridSearchDecisionTree", "tree__"
        )
        base_params = {
            "max_depth": 6,
            "min_samples_leaf": 20,
            "min_samples_split": 2,
        }
        base_params.update(params)
        return DecisionTreeRegressor(
            random_state=42,
            **base_params,
        )


class RandomForestModel(BaseModel):
    def __init__(self, data_path=None, target: str | None = None, use_defaults: bool = False):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)

    def build_model(self):
        if self.use_defaults:
            return RandomForestRegressor(random_state=42, n_jobs=-1)

        params = _load_gridsearch_params(
            self.target, "GridSearchRandomForest", "rf__"
        )
        base_params = {
            "max_depth": 15,
            "min_samples_leaf": 4,
            "n_estimators": 100,
            "n_jobs": -1,
        }
        base_params.update(params)
        return RandomForestRegressor(
            random_state=42,
            **base_params,
        )


class ExtraTreesModel(BaseModel):
    def __init__(self, data_path=None, target: str | None = None, use_defaults: bool = False):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)

    def build_model(self):
        if self.use_defaults:
            return ExtraTreesRegressor(random_state=42, n_jobs=-1)

        params = _load_gridsearch_params(
            self.target, "GridSearchExtraTrees", "et__"
        )
        base_params = {
            "n_estimators": 100,
            "min_samples_leaf": 4,
            "max_features": 1.0,
            "n_jobs": -1,
        }
        base_params.update(params)
        return ExtraTreesRegressor(
            random_state=42,
            **base_params,
        )


class GradientBoostingModel(BaseModel):
    def __init__(self, data_path=None, target: str | None = None, use_defaults: bool = False):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)

    def build_model(self):
        if self.use_defaults:
            return GradientBoostingRegressor(random_state=42)

        params = _load_gridsearch_params(
            self.target, "GridSearchGradientBoosting", "gb__"
        )
        base_params = {
            "learning_rate": 0.09,
            "max_depth": 2,
            "n_estimators": 900,
        }
        base_params.update(params)
        return GradientBoostingRegressor(
            random_state=42,
            **base_params,
        )


class LinearRegressionModel(BaseModel):
    def __init__(self, data_path=None, target: str | None = None, use_defaults: bool = False):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)

    def build_model(self):
        return LinearRegression()


class KNNModel(BaseModel):
    def __init__(
        self,
        n_neighbors: int = 9,
        data_path=None,
        target: str | None = None,
        use_defaults: bool = False,
    ):
        super().__init__(data_path=data_path, target=target, use_defaults=use_defaults)
        self.n_neighbors = n_neighbors

    def build_model(self):
        if self.use_defaults:
            knn = KNeighborsRegressor()
            return Pipeline([
                ('scaler', StandardScaler()),
                ('knn', knn),
            ])

        params = _load_gridsearch_params(
            self.target, "GridSearchKNN", "knn__"
        )
        base_params = {
            "n_neighbors": self.n_neighbors,
            "weights": "distance",
            "p": 1,
            "n_jobs": -1,
        }
        base_params.update(params)
        knn = KNeighborsRegressor(**base_params)
        if params:
            knn.set_params(**params)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', knn),
        ])
