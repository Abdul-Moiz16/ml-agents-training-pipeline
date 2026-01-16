from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from BaselineModel import BaseModel

class GridSearchRandomForest(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'rf__n_estimators': [100, 200, 500],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_leaf': [1, 2, 4]
        }
        return GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)

    def run_optimization(self):
        X, y = self.load_and_encode()
        grid = self.build_model()
        grid.fit(X, y)
        print(f"Best RF Params: {grid.best_params_}")
        return grid.best_params_
