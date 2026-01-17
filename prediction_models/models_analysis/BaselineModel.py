"""
Base preprocessing and evaluation utilities for regression models on main.csv.

- Drops ID/leakage columns.
- Coerces target to numeric and drops any rows with missing values.
- One-hot encodes categoricals for model consumption.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel:

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path(__file__).resolve().parents[3] / "training_manager" / "experiments" / "results" / "main.csv"

    def build_model(self):
        """Override in subclasses to return an instantiated sklearn regressor."""
        raise NotImplementedError


    def cross_validate_with_scaling(self, X, y, n_splits: int = 5, seed: int = 42, transform_log: bool = False) -> dict:

        model = self.build_model()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        if transform_log:
            def exp_mean_absolute_error(y_true_log, y_pred_log):
                return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

            def exp_median_absolute_error(y_true_log, y_pred_log):
                return median_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

            def exp_r2_score(y_true_log, y_pred_log):
                return r2_score(np.expm1(y_true_log), np.expm1(y_pred_log))

            mean_ae_scorer = make_scorer(exp_mean_absolute_error, greater_is_better=False)
            median_ae_scorer = make_scorer(exp_median_absolute_error, greater_is_better=False)
            r2_scorer = make_scorer(exp_r2_score, greater_is_better=True)

            scoring = {
                'mean_ae': mean_ae_scorer,
                'median_ae': median_ae_scorer,
                'r2': r2_scorer
            }

        else:
            scoring = {
                'mean_ae': 'neg_mean_absolute_error',
                'median_ae': 'neg_median_absolute_error',
                'r2': 'r2'
            }


        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "mean_ae": -cv_results['test_mean_ae'].mean(),
            "std_mean_ae": cv_results['test_mean_ae'].std(),
            "mean_r2": cv_results['test_r2'].mean(),
            "std_r2": cv_results['test_r2'].std(),
            "median_ae": -cv_results['test_median_ae'].mean(),
            "std_median_ae": cv_results['test_median_ae'].std(),
        }
