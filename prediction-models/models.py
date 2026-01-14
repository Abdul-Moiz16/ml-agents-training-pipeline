from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer


from BaselineModel import BaseModel


class DecisionTreeModel(BaseModel):
    def build_model(self):
        return DecisionTreeRegressor(random_state=42)


class RandomForestModel(BaseModel):
    def build_model(self):
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


class ExtraTreesModel(BaseModel):
    def build_model(self):
        return ExtraTreesRegressor ( n_estimators=600, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1)


class GradientBoostingModel(BaseModel):
    def build_model(self):
        return GradientBoostingRegressor(random_state=42)


class LinearRegressionModel(BaseModel):
    def build_model(self):
        return LinearRegression()


class KNNModel(BaseModel):
    def __init__(self, n_neighbors: int = 3, data_path=None):
        super().__init__(data_path)
        self.n_neighbors = n_neighbors

    def build_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                weights='distance',
                n_jobs=-1
            ))
        ])
