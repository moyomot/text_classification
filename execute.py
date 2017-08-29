from data_helpers import AgNews, YahooAnswers, YelpReviewPolarity
from classifiers.cnn_classifier import CNNClassifier
from classifiers.lstm_classifier import LSTMClassifier
from classifiers.character_level_cnn_classifier import CharacterLevelCNNClassifier
from classifiers.naive_bayes_classifier import NaiveBayesClassifier
from classifiers.svm_classifier import SVMClassifier
import argparse

datasets = {
    'ag_news': AgNews(),
    'yahoo_answers': YahooAnswers(),
    'yelp_review_polarity': YelpReviewPolarity(),
}

classifiers = {
    'cnn': CNNClassifier(),
    'lstm': LSTMClassifier(),
    'character_level_cnn': CharacterLevelCNNClassifier(),
    'naive_bayes': NaiveBayesClassifier(),
    'svm': SVMClassifier()
}


def main(classifier_str, dataset_str):
    dataset = datasets[dataset_str]
    classifier = classifiers[classifier_str]
    classifier.load(dataset)
    classifier.fit()
    classifier.evaluate()

    # TODO grid search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cnn')
    parser.add_argument('--dataset', type=str, default='ag_news')

    args = parser.parse_args()

    main(args.method, args.dataset)
