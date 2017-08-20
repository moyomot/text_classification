import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class AgNewsData:
    MAX_NB_WORDS = 200000
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 1000

    @classmethod
    def load(cls):
        word2vec = KeyedVectors.load_word2vec_format('dataset/embedding/GoogleNews-vectors-negative300.bin', binary=True)

        column_names = ['category_id', 'title', 'description']
        with open("dataset/ag_news_csv/train.csv", "r") as file:
            df_train = pd.read_csv(file, names=column_names, header=None)
        with open("dataset/ag_news_csv/test.csv", "r") as file:
            df_test = pd.read_csv(file, names=column_names, header=None)

        X_texts = list(df_train.title + df_train.description)
        y_labels = list(df_train.category_id)


        val_X_texts = list(df_test.title + df_test.description)
        val_y_labels = list(df_test.category_id)


        tokenizer = Tokenizer(num_words=cls.MAX_NB_WORDS)
        tokenizer.fit_on_texts(X_texts + val_X_texts)
        X_sequences = tokenizer.texts_to_sequences(X_texts)
        val_X_sequences = tokenizer.texts_to_sequences(val_X_texts)

        word_index = tokenizer.word_index

        X = pad_sequences(X_sequences, maxlen=cls.MAX_SEQUENCE_LENGTH)
        val_X = pad_sequences(val_X_sequences, maxlen=cls.MAX_SEQUENCE_LENGTH)

        y_labels = to_categorical(np.asarray(y_labels))
        val_y_labels = to_categorical(np.asarray(val_y_labels))
