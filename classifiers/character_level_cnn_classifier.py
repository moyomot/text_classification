"""
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
This is not a same model.
"""
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D


from logs import logger


class CharacterLevelCNNClassifier:
    def load(self, dataset):
        logger.info("[character level cnn classifier start loading dataset]")
        dataset.create_character_level_dataset()
        self.dataset = dataset

    def fit(self):
        dense_outputs = 1024
        logger.info("[character level cnn classifier start creating model]")
        input = Input(shape=(self.dataset.X_train.shape[1], self.dataset.X_train.shape[2]), dtype='int8')
        x = Conv1D(256, 7, activation='relu')(input)
        x = MaxPooling1D(3)(x)
        x = Conv1D(256, 7, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(Dense(dense_outputs, activation='relu')(x))
        z = Dense(128, activation='relu')(x)
        preds = Dense(self.dataset.category_size, activation='softmax')(z)
        model = Model(input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        logger.info(self.dataset.X_train.shape)
        logger.info(self.dataset.X_test.shape)
        model.fit(self.dataset.X_train,
                  self.dataset.y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(self.dataset.X_test, self.dataset.y_test),
                  callbacks=[early_stopping])
        self.model = model

    def evaluate(self):
        logger.info("[character level cnn classifier start evaluating model]")
        score = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=0)
        logger.info(score)
