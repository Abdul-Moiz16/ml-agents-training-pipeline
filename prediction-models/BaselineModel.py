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
from sklearn.model_selection import KFold, train_test_split


class BaseModel:
    
    target_col: str = "time_to_convergence"
    drop_cols: List[str] = [
        "run_id",
        "run_log_file",
        "train_duration_s",     # leakage
        "steps_to_convergence"  # leakage
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

    def cross_validate_model(self, n_splits: int = 5, seed: int = 42) -> dict:
        """K-fold CV evaluation; returns mean MAE and mean R^2."""
        X, y = self.load_and_encode()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        maes = []
        r2s = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = self.build_model()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            r2s.append(r2_score(y_test, preds))

        return {
            "mean_mae": sum(maes) / len(maes) if maes else float("nan"),
            "mean_r2": sum(r2s) / len(r2s) if r2s else float("nan"),
        }

    def save_encoded(self, out_path: Optional[Path] = None) -> Path:
        """Persist the encoded feature matrix + target for inspection."""
        
        X, y = self.load_and_encode()
        
        encoded = X.copy()
        encoded[self.target_col] = y
        
        out = out_path or (Path(__file__).resolve().parent / "encoded_dataset.csv")
        encoded.to_csv(out, index=False)
        
        return out
