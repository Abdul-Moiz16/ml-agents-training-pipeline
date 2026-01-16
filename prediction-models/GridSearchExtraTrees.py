from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from BaselineModel import BaseModel

class GridSearchExtraTrees(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('et', ExtraTreesRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'et__n_estimators': [200, 400, 600],
            'et__max_features': [0.8, 'sqrt', 1.0],
            'et__min_samples_leaf': [1, 2, 4]
        }
        return GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)

    def run_optimization(self):
        X, y = self.load_and_encode()
        grid = self.build_model()
        grid.fit(X, y)
        print(f"Best ET Params: {grid.best_params_}")
        return grid.best_params_
