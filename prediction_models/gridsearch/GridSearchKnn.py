import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_models.models_analysis.BaselineModel import BaseModel

class GridSearchKNN(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_jobs=-1))
        ])
        param_grid = {
            'knn__n_neighbors':[3, 5, 7, 9, 11, 15, 21, 25, 31, 34, 38],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2] # 1: Manhattan, 2: Euclidean
        }
        return GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)

    def run_optimization(self):
        X, y = self.load_and_encode()

        print(f"--- Starting Grid Search for {self.__class__.__name__} ---")
        grid = self.build_model()
        grid.fit(X, y)

        print("\n" + "=" * 40)
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV MAE: {-grid.best_score_:.4f}")
        print("=" * 40)

        return grid.best_params_
