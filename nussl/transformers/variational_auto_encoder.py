from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential, load_model
from keras import optimizers
from sklearn.base import TransformerMixin
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import objectives


class VariationalAutoEncoder(TransformerMixin):
    def __init__(self, encoding_dim=256, latent_dim=10, input_shape=1025, output_shape=1025, activation_sparsity=0.01,
                 template_sparsity=1, loss='vae', optimizer=optimizers.rmsprop(lr=.001)):
        self.encoding_dim = encoding_dim
        self.loss = self.vae_loss
        self.optimizer = optimizer
        self.latent_dim = latent_dim

        input_frame = Input(shape=(input_shape,), name='input')
        self.encoded = Dense(self.encoding_dim, activation='softplus',
                            activity_regularizer=regularizers.l1(activation_sparsity),
                             name='encoder')(input_frame)

        self.z_mean = Dense(self.latent_dim)(self.encoded)
        self.z_mu = Dense(self.latent_dim)(self.encoded)

        self.z = Lambda(self.sampling)([self.z_mean, self.z_mu])
        self.decoded = Dense(self.encoding_dim, activation='softplus')(self.z)
        self.decoded = Dense(output_shape, activation='softplus',
                             kernel_regularizer=regularizers.l2(template_sparsity),
                             name='decoder')(self.z)

        self.autoencoder = Model(input_frame, self.decoded)
        #self.encoder = Model(input_frame, self.z)
        #self.decoder = Model(self.z, self.decoded)

        self.has_fit_been_run = False

        self.autoencoder.compile(loss=self.loss, optimizer=self.optimizer)

    def sampling(self, args):
        z_mean, z_mu = args
        epsilon = K.random_normal(shape=(self.latent_dim,), mean=0., stddev=1.)
        return z_mean + K.exp(z_mu / 2.) * epsilon



    def fit(self, input_data, output_data, validation_data=None, epochs=50, batch_size=256, shuffle=True, callbacks=None):
        self.autoencoder.fit(input_data, output_data,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=validation_data,
                             callbacks=callbacks)
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
        self.autoencoder = load_model(path, custom_objects={'vae_loss': self.vae_loss,
                                                            'sampling': self.sampling})
        self.has_fit_been_run = True
        return self

    def vae_loss(self, y_true, y_pred):
        mse = objectives.mean_squared_error(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + self.z_mu - K.square(self.z_mean) - K.exp(self.z_mu), axis=-1)
        return mse + kl_loss

    def kl_divergence_nmf(self, y_true, y_pred):
        return K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred)

    def error_measure(self, y_true, y_pred, axis = None):
        # y_true = y_true.astype(dtype=np.float64)
        # y_pred = y_pred.astype(dtype=np.float64)
        # return K.eval(K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred))
        return np.sum(np.multiply(y_true, (np.log(y_true + K.epsilon()) - np.log(y_pred + K.epsilon()))) - y_true + y_pred, axis = axis) \
               / float(y_true.shape[0])
        # return np.linalg.norm(np.abs(y_true - y_pred), 'fro', axis=axis) / float(y_true.shape[0])

