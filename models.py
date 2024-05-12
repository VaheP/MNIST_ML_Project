from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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


class LogisticRegressionModel(Model):
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs'):
        super().__init__()
        self.model = LogisticRegression(C=C, penalty=penalty, solver=solver)

    def fit(self, X, y):
        self.model.fit(X, y)


class KNNModel(Model):
    def __init__(self, n_neighbors=5, leaf_size=30, weights='uniform'):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)

    def fit(self, X, y):
        self.model.fit(X, y)


class SVMModel(Model):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(self, X, y):
        self.model.fit(X, y)


