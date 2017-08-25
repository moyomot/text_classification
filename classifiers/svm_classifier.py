import numpy as np
from sklearn.svm import LinearSVC

class SVMClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load(self, dataset):
        dataset.create_tfidf_dataset()
        self.X_train_tfidf = dataset.X_train_tfidf
        self.y_train_labels = dataset.y_train_labels
        self.X_test_tfidf = dataset.X_test_tfidf
        self.y_test_labels = dataset.y_test_labels

    def fit(self):
        self.clf = LinearSVC().fit(self.X_train_tfidf, self.y_train_labels)

    def predict(self):
        predicted = self.clf.predict(self.X_test_tfidf)
        print(np.mean(predicted==self.y_test_labels))
