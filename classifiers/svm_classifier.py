import numpy as np
from sklearn.svm import LinearSVC

from logs import logger


class SVMClassifier:
    def load(self, dataset):
        logger.info("[svm classifier start loading dataset]")
        dataset.create_tfidf_dataset()
        self.dataset = dataset

    def fit(self):
        logger.info("[svm classifier start creating model]")
        self.clf = LinearSVC().fit(self.dataset.X_train_tfidf, self.dataset.y_train_labels)

    def evaluate(self):
        logger.info("[svm classifier start evaluating model]")
        predicted = self.clf.predict(self.dataset.X_test_tfidf)
        logger.info(np.mean(predicted==self.dataset.y_test_labels))
