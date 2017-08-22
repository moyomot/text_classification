from keras.layers import Bidirectional, Dense, Input, Embedding, LSTM
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


class LSTMClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.embedding_matrix = None
        self.word_index = None

    def load(self, dataset):
        dataset.create_cnn_dataset()
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
        l_lstm = Bidirectional(LSTM(100))(embedded_sequences)

        preds = Dense(2, activation='softmax')(l_lstm)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.summary()
        model.fit(self.X_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(self.X_test, self.y_test))
