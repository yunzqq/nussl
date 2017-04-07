from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import optimizers
from sklearn.base import TransformerMixin
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def loss_function(y_true, y_pred):
    return K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred)

class VariationalAutoEncoder(TransformerMixin):
    def __init__(self, encoding_dim=256, input_shape=1025, output_shape=1025, activation_sparsity=0.01,
                 template_sparsity=1, loss=loss_function, optimizer=optimizers.rmsprop(lr=.001)):
        encoding_dim = encoding_dim
        self.loss = loss
        self.optimizer = optimizer

        input_frame = Input(shape=(input_shape,))
        self.encoded = Dense(encoding_dim, activation='softplus',
                            activity_regularizer=regularizers.l1(activation_sparsity))(input_frame)
        self.decoded = Dense(output_shape, activation='softplus',
                             kernel_regularizer=regularizers.l2(template_sparsity))(self.encoded)

        self.autoencoder = Model(input_frame, self.decoded)


        decoder_layer = self.autoencoder.layers[-1]

        self.encoder = Model(input_frame, self.encoded)

        encoded_input = Input(shape=(encoding_dim,))
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        self.has_fit_been_run = False
        self.has_transform_been_run = False

        self.autoencoder.compile(loss=self.loss, optimizer=self.optimizer)

    def fit(self, input_data, output_data, validation_data=None, epochs=50, batch_size=256, shuffle=True, callbacks=None):
        self.autoencoder.fit(input_data, output_data,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=validation_data)
        plt.figure()
        plt.imshow(self.autoencoder.get_weights()[-2].T, aspect='auto')
        plt.show()
        self.has_fit_been_run = True
        return self

    def transform(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        self.representation = self.encoder.predict(X)
        return self.representation

    def reconstruction_error(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        reconstruction = self.inverse_transform(X)
        return np.linalg.norm(X - reconstruction, 'fro')

    def reconstruction_error_by_frame(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")

        reconstruction = self.inverse_transform(X)
        return np.linalg.norm(X - reconstruction, axis=1)

    def inverse_transform(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        reconstruction = self.autoencoder.predict(X)
        return reconstruction