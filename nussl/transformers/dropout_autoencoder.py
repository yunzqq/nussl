from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.base import TransformerMixin
from keras import regularizers
import numpy as np

class DropoutAutoEncoder(TransformerMixin):
    def __init__(self, encoding_dim=256, input_shape=1025, output_shape=1025, sparsity=10e-5, dropout_amount=.5):
        encoding_dim = encoding_dim

        input_frame = Input(shape=(input_shape,))
        encoded = Dense(encoding_dim, activation="relu",
                        activity_regularizer=regularizers.l1(sparsity))(input_frame)
        dropout = Dropout(dropout_amount)(encoded)
        decoded = Dense(output_shape, activation="relu")(dropout)

        self.autoencoder = Model(input_frame, decoded)
        self.encoder = Model(input_frame, encoded)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.autoencoder.compile(loss='mean_squared_error',
                                 optimizer='rmsprop')

    def fit(self, _in, _out, validation_data=None, epochs=50, batch_size=256, shuffle=True, callbacks=None):
        self.autoencoder.fit(_in, _out,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=validation_data)
        return self

    def transform(self, X):
        predictions = self.encoder.predict(X)
        self.representation = predictions
        return predictions

    def reconstruction_error(self, X):
        reconstruction = self.inverse_transform(self.transform(X))
        return np.linalg.norm(X - reconstruction, 'fro')

    def reconstruction_error_by_frame(self, X):
        reconstruction = self.inverse_transform(self.transform(X))
        return np.linalg.norm(X - reconstruction, axis=1)

    def inverse_transform(self, X):
        reconstruction = self.decoder.predict(X)
        return reconstruction