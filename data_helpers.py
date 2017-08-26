import re

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from logs import logger

WORD2VEC_PATH = 'dataset/embedding/GoogleNews-vectors-negative300.bin'


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


class DataSet:
    @classmethod
    def load_word2vec(cls):
        try:
            word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        except Exception as e:
            raise e
        return word2vec

    def load(self, column_names):
        with open(self.TRAIN_PATH, "r") as file:
            self.df_train = pd.read_csv(file, names=column_names, header=None)
        with open(self.TEST_PATH, "r") as file:
            self.df_test = pd.read_csv(file, names=column_names, header=None)

    def tfidf_transformer(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.X_train)
        tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        self.y_train_labels = self.y_train
        logger.info('train text size is {size}'.format(size=len(self.X_train)))
        logger.info('train label size is {size}'.format(size=len(self.y_train_labels)))

        X_test_counts = count_vect.transform(self.X_test)
        self.X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
        self.y_test_labels = self.y_test
        logger.info('test text size is {size}'.format(size=len(self.X_test)))
        logger.info('test label size is {size}'.format(size=len(self.y_test_labels)))

    def embedding_transfomer(self):
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.X_train + self.X_test)
        X_train_sequences = tokenizer.texts_to_sequences(self.X_train)
        X_test_sequences = tokenizer.texts_to_sequences(self.X_test)

        self.X_train = pad_sequences(X_train_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        self.X_test = pad_sequences(X_test_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

        self.y_train = to_categorical(np.asarray(self.y_train))
        self.y_test = to_categorical(np.asarray(self.y_test))

        word2vec = DataSet.load_word2vec()
        self.word_index = tokenizer.word_index
        embedding_matrix = np.zeros((len(self.word_index), self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
        self.embedding_matrix = embedding_matrix


class AgNews(DataSet):
    TRAIN_PATH = 'dataset/ag_news_csv/train.csv'
    TEST_PATH = 'dataset/ag_news_csv/test.csv'
    COLUMN_NAMES = ['category_id', 'title', 'description']

    def __init__(self):
        self.MAX_NB_WORDS = 200000
        self.EMBEDDING_DIM = 300
        self.MAX_SEQUENCE_LENGTH = 1000

    def create_embedding_dataset(self):
        self.load(self.COLUMN_NAMES)
        self.X_train = list(self.df_train.title + self.df_train.description)
        self.X_train = [clean_str(text) for text in self.X_train]
        self.y_train = list(self.df_train.category_id)
        self.X_test = list(self.df_test.title + self.df_test.description)
        self.X_test = [clean_str(text) for text in self.X_test]
        self.y_test = list(self.df_test.category_id)
        self.embedding_transfomer()

    def create_tfidf_dataset(self):
        self.load(self.COLUMN_NAMES)
        self.X_train = list(self.df_train.title + self.df_train.description)
        self.X_train = [clean_str(text) for text in self.X_train]
        self.y_train = list(self.df_train.category_id)
        self.X_test = list(self.df_test.title + self.df_test.description)
        self.X_test = [clean_str(text) for text in self.X_test]
        self.y_test = list(self.df_test.category_id)
        self.tfidf_transformer()


class YahooAnswers(DataSet):
    TRAIN_PATH = 'dataset/yahoo_answers_csv/train.csv'
    TEST_PATH = 'dataset/yahoo_answers_csv/test.csv'
    COLUMN_NAMES = ['category_id', 'title', 'question', 'answer']

    def __init__(self):
        self.MAX_NB_WORDS = 1200000
        self.EMBEDDING_DIM = 300
        self.MAX_SEQUENCE_LENGTH = 2000

    def create_tfidf_dataset(self):
        self.load(self.COLUMN_NAMES)
        self.X_train = list(self.df_train.title.fillna(" ") + self.df_train.question.fillna(" ") + self.df_train.answer.fillna(" "))
        self.X_train = [clean_str(text) for text in self.X_train]
        self.y_train = list(self.df_train.category_id)
        self.X_test = list(self.df_test.title.fillna(" ") + self.df_test.question.fillna(" ") + self.df_test.answer.fillna(" "))
        self.X_test = [clean_str(text) for text in self.X_test]
        self.y_test = list(self.df_test.category_id)
        self.tfidf_transformer()

    def create_embedding_dataset(self):
        self.load(self.COLUMN_NAMES)
        self.X_train = list(self.df_train.title.fillna(" ") + self.df_train.question.fillna(" ") + self.df_train.answer.fillna(" "))
        self.X_train = [clean_str(text) for text in self.X_train]
        self.y_train = list(self.df_train.category_id)
        self.X_test = list(self.df_test.title.fillna(" ") + self.df_test.question.fillna(" ") + self.df_test.answer.fillna(" "))
        self.X_test = [clean_str(text) for text in self.X_test]
        self.y_test = list(self.df_test.category_id)
        self.embedding_transfomer()
