"""
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
"""
from keras.layers import Dense, Flatten, Input, Embedding
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping

from logs import logger


class CNNClassifier:
    def load(self, dataset):
        logger.info("[cnn classifier start loading dataset]")
        dataset.create_embedding_dataset()
        self.dataset = dataset

    def fit(self):
        logger.info("[cnn classifier start creating model]")
        num_words = min(self.dataset.MAX_NB_WORDS, len(self.dataset.word_index))
        embedding_layer = Embedding(num_words,
                                    self.dataset.EMBEDDING_DIM,
                                    weights=[self.dataset.embedding_matrix],
                                    input_length=self.dataset.MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        sequence_input = Input(shape=(self.dataset.MAX_SEQUENCE_LENGTH,), dtype='int32')
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
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model.fit(self.dataset.X_train,
                  self.dataset.y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(self.dataset.X_test, self.dataset.y_test),
                  callbacks=[early_stopping])
        self.model = model

    def evaluate(self):
        logger.info("[cnn classifier start evaluating model]")
        score = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        logger.info(score)
