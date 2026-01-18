from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from .BaselineModel import BaseModel
except ImportError:
    from BaselineModel import BaseModel


class DecisionTreeModel(BaseModel):
    def build_model(self):
        return DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, min_samples_split=2, random_state=42)


class RandomForestModel(BaseModel):
    def build_model(self):
        return RandomForestRegressor(max_depth=8, min_samples_leaf=15, n_estimators=600, random_state=42, n_jobs=-1)


class ExtraTreesModel(BaseModel):
    def build_model(self):
        return ExtraTreesRegressor ( n_estimators=300, min_samples_leaf=15, max_features=1.0, random_state=42, n_jobs=-1)


class GradientBoostingModel(BaseModel):
    def build_model(self):
        return GradientBoostingRegressor(learning_rate=0.1, max_depth=2, n_estimators=200, random_state=42)


class LinearRegressionModel(BaseModel):
    def build_model(self):
        return LinearRegression()


class KNNModel(BaseModel):
    def __init__(self, n_neighbors: int = 9, data_path=None):
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
