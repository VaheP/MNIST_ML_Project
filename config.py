# Configuration settings
DATA_PATH = 'data/fashion-mnist_train.csv'
DATA_PATH_TEST = 'data/fashion-mnist_test.csv'
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs'
    },
    'knn': {
        'n_neighbors': 5,
        'leaf_size': 30,
        'weights': 'uniform'
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
}
