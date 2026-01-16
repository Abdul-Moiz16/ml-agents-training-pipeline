"""
Base preprocessing and evaluation utilities for regression models on main.csv.

- Drops ID/leakage columns.
- Coerces target to numeric and drops any rows with missing values.
- One-hot encodes categoricals for model consumption.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel:

    target_col: str = "time_to_convergence"
    drop_cols: List[str] = [
        "run_id",
        "run_log_file",
        "train_duration_s",
        "steps_to_convergence",
        # Following features added by Pawel since unavailable pre-run
        "avg_cpu_usage",
        "avg_ram_usage",
        "peak_ram_usage",
        "peak_cpu_usage",
        "final_std_reward",
        "final_mean_reward"
    ]

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path(__file__).resolve().parents[1] / "training_manager" / "experiments" / "results" / "main.csv"

    def load_and_encode(self) -> Tuple[pd.DataFrame, pd.Series]:

        df = pd.read_csv(self.data_path)

        df = df.drop(columns=self.drop_cols, errors="ignore")

        df = df.replace("NA", pd.NA)
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df = df.dropna()

        # One-hot encode remaining categorical/string columns so the regressor can consume them.
        X = df.drop(columns=[self.target_col], errors="ignore")
        y = df[self.target_col]

        X = pd.get_dummies(X, drop_first=True)

        return X, y

    def build_model(self):
        """Override in subclasses to return an instantiated sklearn regressor."""

        raise NotImplementedError

    # Evaluate the model using Mean Absolute Error (Lower is Better) and R^2 Score (Closer to 1 is Better)
    def train_test_eval(self, test_size: float = 0.2, seed: int = 42) -> Tuple[float, float, object]:
        """Train/test split evaluation; returns MAE, R^2, and the fitted model."""

        X, y = self.load_and_encode()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        model = self.build_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        return mae, r2, model

    def cross_validate_with_scaling(self, n_splits: int = 5, seed: int = 42) -> dict:

        X, y = self.load_and_encode()
        model = self.build_model()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        scoring = {
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }


        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "mean_mae": -cv_results['test_mae'].mean(),
            "std_mae": cv_results['test_mae'].std(),
            "mean_r2": cv_results['test_r2'].mean(),
            "std_r2": cv_results['test_r2'].std()
        }

    def save_encoded(self, out_path: Optional[Path] = None) -> Path:
        """Persist the encoded feature matrix + target for inspection."""

        X, y = self.load_and_encode()

        encoded = X.copy()
        encoded[self.target_col] = y

        base_dir = Path(__file__).resolve().parent
        out = out_path or (base_dir / "encoded_dataset.csv")
        encoded.to_csv(out, index=False)

        feature_sel_dir = base_dir / "feature-selection"
        feature_sel_dir.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(feature_sel_dir / "encoded_dataset.csv", index=False)
        
        return out
