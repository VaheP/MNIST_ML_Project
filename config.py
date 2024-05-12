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
}
