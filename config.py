# Configuration settings
DATA_PATH = 'data/fashion-mnist_train.csv'
DATA_PATH_TEST = 'data/fashion-mnist_test.csv'
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'logistic_regression': {
        'C': 10,
        'penalty': 'l2',
        'solver': 'newton-cg'
    },
    'knn': {
        'n_neighbors': 5,
        'leaf_size': 15,
        'weights': 'distance'
    },
    'svm': {
        'C': 10,
        'kernel': 'rbf',
        'gamma': 0.01
    },
    'bagging': {
        'n_estimators': 50,
        'n_neighbors': 5,
        'leaf_size': 45,
        'weights': 'distance',
        'bootstrap': True,
    },
    'adaboost': {
        'n_estimators': 50,
        'learning_rate': 0.5
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.5,
        'gamma': 0.1
    }
}
