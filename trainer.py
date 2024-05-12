from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        # Use 'weighted' for imbalanced datasets
        average_type = 'macro'
        return {
            'precision': precision_score(y_test, predictions, average=average_type),
            'recall': recall_score(y_test, predictions, average=average_type),
            'f1': f1_score(y_test, predictions, average=average_type),
            'accuracy': accuracy_score(y_test, predictions),
            'roc_auc': roc_auc_score(label_binarize(y_test, classes=np.unique(y_test)),
                                     label_binarize(predictions, classes=np.unique(y_test)),
                                     average=average_type, multi_class='ovr')
        }
