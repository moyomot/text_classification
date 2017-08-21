import logging

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

logger = logging.getLogger(__name__)


WORD2VEC_PATH = 'dataset/embedding/GoogleNews-vectors-negative300.bin'
AG_NEWS_TRAIN_PATH = 'dataset/ag_news_csv/train.csv'
AG_NEWS_TEST_PATH = 'dataset/ag_news_csv/test.csv'


class DataSet:
    @classmethod
    def load_word2vec(cls):
        try:
            word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        except Exception as e:
            raise e
        return word2vec


class AgNews(DataSet):
    MAX_NB_WORDS = 200000
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 1000

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_train = None
        self.df_test = None
        self.embedding_matrix = None

    def load(self):
        column_names = ['category_id', 'title', 'description']
        with open(AG_NEWS_TRAIN_PATH, "r") as file:
            self.df_train = pd.read_csv(file, names=column_names, header=None)
        with open(AG_NEWS_TEST_PATH, "r") as file:
            self.df_test = pd.read_csv(file, names=column_names, header=None)

    def create_cnn_dataset(self):
        self.load()
        X_train_texts = list(self.df_train.title + self.df_train.description)
        y_train_labels = list(self.df_train.category_id)
        logger.info('train text size is '.format(len(X_train_texts)))
        logger.info('train label size is'.format(len(y_train_labels)))

        X_test_texts = list(self.df_test.title + self.df_test.description)
        y_test_labels = list(self.df_test.category_id)

        tokenizer = Tokenizer(num_words=AgNews.MAX_NB_WORDS)
        tokenizer.fit_on_texts(X_train_texts + X_test_texts)
        X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
        X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)

        self.X_train = pad_sequences(X_train_sequences, maxlen=AgNews.MAX_SEQUENCE_LENGTH)
        self.X_test = pad_sequences(X_test_sequences, maxlen=AgNews.MAX_SEQUENCE_LENGTH)

        self.y_train = to_categorical(np.asarray(y_train_labels))
        self.y_test = to_categorical(np.asarray(y_test_labels))

        word2vec = DataSet.load_word2vec()
        self.word_index = tokenizer.word_index
        embedding_matrix = np.zeros((len(self.word_index), AgNews.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
        self.embedding_matrix = embedding_matrix
