import os
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from BaselineModel import BaseModel

class GridSearchDecisionTree(BaseModel):
    """
    Independent class to optimize Decision Tree hyperparameters.
    Ensures scaling happens inside each fold to prevent leakage.
    """

    def build_model(self):
        # 1. Define the Pipeline: Scaling then Regressor
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', DecisionTreeRegressor(random_state=42))
        ])

        # 2. Define the Parameter Grid
        # NOTE: Use 'tree__' prefix to match the pipeline step name
        param_grid = {
            'tree__max_depth': [None, 5, 10, 20],
            'tree__min_samples_split': [2, 5, 10],
            'tree__min_samples_leaf': [1, 2, 4]
        }

        # 3. Setup GridSearchCV
        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=5,  # 5-fold internal cross-validation
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )

    def run_optimization(self):
        """Runs the search on the dataset and prints best values."""
        X, y = self.load_and_encode()

        print(f"--- Starting Grid Search for {self.__class__.__name__} ---")
        grid_search = self.build_model()
        grid_search.fit(X, y)

        print("\n" + "=" * 40)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: {-grid_search.best_score_:.4f}")
        print("=" * 40)

        return grid_search.best_params_


if __name__ == "__main__":
    optimizer = GridSearchDecisionTree()
    optimizer.run_optimization()
