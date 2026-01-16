from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from BaselineModel import BaseModel

class GridSearchKNN(BaseModel):
    def build_model(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_jobs=-1))
        ])
        param_grid = {
            'knn__n_neighbors': [3, 5, 11, 21],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2] # 1: Manhattan, 2: Euclidean
        }
        return GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)

    def run_optimization(self):
        X, y = self.load_and_encode()
        grid = self.build_model()
        grid.fit(X, y)
        print(f"Best KNN Params: {grid.best_params_}")
        return grid.best_params_
