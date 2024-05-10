from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        # return precision, recall, f1, accuracy, roc_auc
        return {
            'precision': precision_score(y_test, self.model.predict(X_test)),
            'recall': recall_score(y_test, self.model.predict(X_test)),
            'f1': f1_score(y_test, self.model.predict(X_test)),
            'accuracy': accuracy_score(y_test, self.model.predict(X_test)),
            'roc_auc': roc_auc_score(y_test, self.model.predict(X_test))
        }
