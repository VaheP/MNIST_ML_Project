import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, file_path, file_path_test):
        self.file_path = file_path
        self.file_path_test = file_path_test
        self.scaler = None
        self.pca = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def load_data(self):
        test_data = None
        if self.file_path_test is not None:
            test_data = pd.read_csv(self.file_path_test)
            self.X_test = test_data.iloc[:, 1:]
            self.y_test = test_data['label']
        self.data = pd.read_csv(self.file_path)
        self.X_train = self.data.iloc[:, 1:]
        self.y_train = self.data['label']

    def preprocess_data(self, pca_count=200):
        # Ensure scaler is initialized and fit
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X_train)

        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Ensure PCA is initialized and fit with the specified number of components
        if self.pca is None:
            self.pca = PCA(n_components=pca_count)
            self.pca.fit(self.X_train)

        self.X_train = self.pca.transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)

    def split_data(self, test_size=0.2, random_state=42):
        if self.X_test is not None and self.y_test is not None and self.X_train is not None and self.y_train is not None:
            return
        features = self.data.iloc[:, :-1]
        target = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
