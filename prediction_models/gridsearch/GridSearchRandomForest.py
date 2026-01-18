import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_models.models_analysis.BaselineModel import BaseModel

class GridSearchRandomForest(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'rf__n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
            'rf__max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
            'rf__min_samples_leaf': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]
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
