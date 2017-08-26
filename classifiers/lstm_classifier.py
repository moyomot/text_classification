from keras.layers import Bidirectional, Dense, Input, Embedding, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping

from logs import logger


class LSTMClassifier:
    def load(self, dataset):
        logger.info("[lstm classifier start loading dataset]")
        dataset.create_embedding_dataset()
        self.dataset = dataset

    def fit(self):
        logger.info("[lstm classifier start creating model]")
        num_words = min(self.dataset.MAX_NB_WORDS, len(self.dataset.word_index))
        embedding_layer = Embedding(num_words,
                                    self.dataset.EMBEDDING_DIM,
                                    weights=[self.dataset.embedding_matrix],
                                    input_length=self.dataset.MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        sequence_input = Input(shape=(self.dataset.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_lstm = Bidirectional(LSTM(100))(embedded_sequences)

        preds = Dense(5, activation='softmax')(l_lstm)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model.fit(self.dataset.X_train,
                  self.dataset.y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(self.dataset.X_test, self.dataset.y_test),
                  callbacks=[early_stopping])
        self.model = model

    def evaluate(self):
        logger.info("[lstm classifier start evaluating model]")
        score = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        logger.info(score)
