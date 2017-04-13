from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras import optimizers
from sklearn.base import TransformerMixin
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


class BasicAutoEncoder(TransformerMixin):
    def __init__(self, encoding_dim=128, input_shape=1025, output_shape=1025, activation_sparsity=0,
                 template_sparsity=0, loss='mean_squared_error', optimizer='rmsprop'):
        if loss == 'kl_divergence_nmf':
            loss = self.kl_divergence_nmf
        encoding_dim = encoding_dim
        self.loss = loss
        self.optimizer = optimizer

        input_frame = Input(shape=(input_shape,), name='input')
        self.encoded = Dense(encoding_dim, activation='softplus',
                            activity_regularizer=regularizers.l1(activation_sparsity),
                             name='encoder')(input_frame)
        self.decoded = Dense(output_shape, activation='softplus',
                             kernel_regularizer=regularizers.l2(template_sparsity),
                             name='decoder')(self.encoded)

        self.autoencoder = Model(input_frame, self.decoded)
        self.encoder = Model(input_frame, self.encoded)

        self.has_fit_been_run = False

        self.autoencoder.compile(loss=self.loss, optimizer=self.optimizer)

    def fit(self, input_data, output_data, **kwargs):
        self.autoencoder.fit(input_data, output_data,
                             **kwargs)
        self.has_fit_been_run = True
        return self

    def transform(self, X):
        raise NotImplementedError("Haven't fgured out this function yet!")

        # if not self.has_fit_been_run:
        #     raise ValueError("Model has not been fit! Run fit() before calling this.")
        # self.representation = self.encoder.predict(X)
        # return self.representation

    def reconstruction_error(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        loss = self.error_measure(X, self.autoencoder.predict(X))
        return loss

    def reconstruction_error_by_frame(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")

        loss = self.error_measure(X, self.autoencoder.predict(X), axis = 0)
        return loss

    def inverse_transform(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        reconstruction = self.autoencoder.predict(X)
        return reconstruction

    def save(self, path):
        self.autoencoder.save(path)

    def load(self, path):
        self.autoencoder = load_model(path, custom_objects={'kl_divergence_nmf': self.kl_divergence_nmf})
        self.has_fit_been_run = True
        return self

    def kl_divergence_nmf(self, y_true, y_pred):
        return K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred)

    def error_measure(self, y_true, y_pred, axis = None):
        # y_true = y_true.astype(dtype=np.float64)
        # y_pred = y_pred.astype(dtype=np.float64)
        # return K.eval(K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred))
        return np.sum(np.multiply(y_true, (np.log(y_true + K.epsilon()) - np.log(y_pred + K.epsilon())))
                      - y_true + y_pred, axis=axis) \
               / float(y_true.shape[0])

