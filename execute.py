from data_helpers import AgNews
from classifiers.cnn_classifier import CNNClassifier
from classifiers.lstm_classifier import LSTMClassifier
import argparse

datasets = {'ag_news': AgNews()}
classifiers = {'cnn': CNNClassifier(), 'lstm': LSTMClassifier()}


def main(classifier_str, dataset_str):
    dataset = datasets[dataset_str]
    classifier = classifiers[classifier_str]
    classifier.load(dataset)
    classifier.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cnn')
    parser.add_argument('--dataset', type=str, default='ag_news')

    args = parser.parse_args()

    main(args.method, args.dataset)
