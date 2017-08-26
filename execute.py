from data_helpers import AgNews, YahooAnswers
from classifiers.cnn_classifier import CNNClassifier
from classifiers.lstm_classifier import LSTMClassifier
from classifiers.naive_bayes_classifier import NaiveBayesClassifier
from classifiers.svm_classifier import SVMClassifier
import argparse

datasets = {'ag_news': AgNews(),
            'yahoo_answers': YahooAnswers()}
classifiers = {'cnn': CNNClassifier(),
               'lstm': LSTMClassifier(),
               'naive_bayes': NaiveBayesClassifier(),
               'svm': SVMClassifier()}


def main(classifier_str, dataset_str):
    dataset = datasets[dataset_str]
    classifier = classifiers[classifier_str]
    classifier.load(dataset)
    classifier.fit()
    classifier.evaluate()

    # TODO shuffle
    # TODO grid search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cnn')
    parser.add_argument('--dataset', type=str, default='ag_news')

    args = parser.parse_args()

    main(args.method, args.dataset)
