import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self, file_path, file_path_test):
        self.file_path = file_path
        self.file_path_test = file_path_test
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def load_data(self):
        test_data = None
        if self.file_path_test is not None:
            test_data = pd.read_csv(self.file_path_test)
            self.X_test = test_data.iloc[:, 1:]
            self.y_test = test_data['label']
        data = pd.read_csv(self.file_path)

        if test_data is not None:
            self.X_train = data.iloc[:, 1:]
            self.y_train = data['label']
            self.data = pd.concat([data, test_data])
        else:
            self.data = pd.read_csv(self.file_path)
    #
    # def preprocess_data(self):
    #     # Preprocess steps (e.g., normalization, encoding)
    #     pass

    def split_data(self, test_size=0.2, random_state=42):
        if self.X_test is not None and self.y_test is not None and self.X_train is not None and self.y_train is not None:
            return
        features = self.data.iloc[:, :-1]
        target = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
