import numpy as np
from sklearn.naive_bayes import MultinomialNB

from logs import logger


class NaiveBayesClassifier:
    def load(self, dataset):
        logger.info("[naive bayes classifier start loading dataset]")
        dataset.create_tfidf_dataset()
        self.dataset = dataset

    def fit(self):
        logger.info("[naive bayes classifier start creating model]")
        self.clf = MultinomialNB().fit(self.dataset.X_train_tfidf, self.dataset.y_train_labels)

    def evaluate(self):
        logger.info("[naive bayes classifier start evaluating model]")
        predicted = self.clf.predict(self.dataset.X_test_tfidf)
        logger.info(np.mean(predicted==self.dataset.y_test_labels))
