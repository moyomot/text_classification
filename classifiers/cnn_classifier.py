"""
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
"""
from keras.layers import Dense, Flatten, Input, Embedding
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


class CNNClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.embedding_matrix = None
        self.word_index = None

    def load(self, dataset):
        dataset.create_embedding_dataset()
        self.X_train = dataset.X_train
        self.y_train = dataset.y_train
        self.X_test = dataset.X_test
        self.y_test = dataset.y_test
        self.embedding_matrix = dataset.embedding_matrix
        self.word_index = dataset.word_index

    def fit(self):
        num_words = min(MAX_NB_WORDS, len(self.word_index))
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        preds = Dense(5, activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.summary()
        model.fit(self.X_train, self.y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(self.X_test, self.y_test))
