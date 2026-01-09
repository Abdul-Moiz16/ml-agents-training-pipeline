from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from BaselineModel import BaseModel


class DecisionTreeModel(BaseModel):
    def build_model(self):
        return DecisionTreeRegressor(random_state=42)


class RandomForestModel(BaseModel):
    def build_model(self):
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


class ExtraTreesModel(BaseModel):
    def build_model(self):
        return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)


class GradientBoostingModel(BaseModel):
    def build_model(self):
        return GradientBoostingRegressor(random_state=42)


class LinearRegressionModel(BaseModel):
    def build_model(self):
        return LinearRegression()
