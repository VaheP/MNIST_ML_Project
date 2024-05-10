from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Model(BaseEstimator):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class RandomForestModel(Model):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)
