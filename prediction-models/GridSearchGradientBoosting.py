from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from BaselineModel import BaseModel

class GridSearchGradientBoosting(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42))
        ])
        param_grid = {
            'gb__n_estimators': [100, 200, 300],
            'gb__learning_rate': [0.01, 0.05, 0.1],
            'gb__max_depth': [3, 5, 8]
        }
        return GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)

    def run_optimization(self):
        X, y = self.load_and_encode()
        grid = self.build_model()
        grid.fit(X, y)
        print(f"Best GB Params: {grid.best_params_}")
        return grid.best_params_
