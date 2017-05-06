from keras.layers import Input, LSTM, Bidirectional, TimeDistributed, Dense, GaussianNoise
from keras.models import Sequential, load_model, Model
from keras.optimizers import Nadam
from sklearn.base import TransformerMixin
import numpy as np
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import unit_norm


class DeepClustering(TransformerMixin):
    def __init__(self, input_shape=(25, 128), num_sources = 2, batch_size = 128, optimizer='nadam'):
        self.loss = self.affinity_k_means
        self.num_timesteps, self.num_features = input_shape
        self.num_recurrent_layers = 4
        self.embedding_dimensions = 20
        self.size_recurrent_layers = 500
        self.dropout = .5
        self.recurrent_dropout = .2
        self.l2_regularization = 1e-6
        self.clipnorm = 200
        self.num_sources = num_sources
        self.batch_size = batch_size

        if optimizer == 'nadam':
            self.optimizer = Nadam(clipnorm=self.clipnorm)
        else:
            self.optimizer = optimizer

        output_shape = self.num_features * self.embedding_dimensions

        input_sequence = Input(shape=input_shape)
        deep_clusterer = Sequential()
        for i in range(self.num_recurrent_layers):
            lstm_layer = LSTM(self.size_recurrent_layers,
               return_sequences=True,
               kernel_regularizer=l2(self.l2_regularization),
               recurrent_regularizer=l2(self.l2_regularization),
               bias_regularizer=l2(self.l2_regularization),
               dropout=self.dropout,
               recurrent_dropout=self.recurrent_dropout)
            if i == 0:
                layer = Bidirectional(lstm_layer, input_shape=input_shape)
            else:
                layer = Bidirectional(lstm_layer)
            deep_clusterer.add(layer)
            #deep_clusterer.add(GaussianNoise(.6))


        embedding_layer = TimeDistributed(Dense(output_shape,
                                                activation='tanh',
                                                kernel_regularizer=l2(self.l2_regularization),
                                                bias_regularizer=l2(self.l2_regularization)),
                                                name = 'embedding')
        deep_clusterer.add(embedding_layer)
        print deep_clusterer.summary()
        deep_clusterer = deep_clusterer(input_sequence)

        self.deep_clusterer = Model(inputs=[input_sequence], outputs=[deep_clusterer])
        self.deep_clusterer.compile(loss = self.affinity_k_means, optimizer = self.optimizer)

    def affinity_k_means(self, Y, V):
        def norm(tensor):
            square_tensor = K.square(tensor)
            frobenius_norm = K.sum(square_tensor, axis=(1, 2))
            frobenius_norm = K.sqrt(frobenius_norm)
            return frobenius_norm

        def dot(x, y):
            return K.batch_dot(x, y)

        def T(x):
            return K.permute_dimensions(x, [0, 2, 1])

        V = K.l2_normalize(K.reshape(V, (self.batch_size, -1, self.embedding_dimensions)), axis = -1)
        Y = K.reshape(Y, (self.batch_size, -1, self.num_sources))

        silence_mask = K.sum(Y, axis=2, keepdims=True)
        V = silence_mask * V

        return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))

    def fit(self, input_data, output_data, **kwargs):
        self.deep_clusterer.fit(input_data, output_data,
                             **kwargs)
        self.has_fit_been_run = True
        return self

    def fit_generator(self, *args, **kwargs):
        self.deep_clusterer.fit_generator(*args, **kwargs)
        self.has_fit_been_run = True
        return self

    def transform(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        if len(X.shape) < 3:
            X = np.expand_dims(X, axis = 0)
        self.representation = self.deep_clusterer.predict(X)
        return self.representation

    def reconstruction_error(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        reconstruction = self.deep_clusterer.predict(X)
        loss = self.error_measure(X, reconstruction[0]) - self.error_measure(X, reconstruction[1])
        return loss

    def reconstruction_error_by_frame(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")

        loss = self.error_measure(X, self.deep_clusterer.predict(X), axis = 0)
        return loss

    def inverse_transform(self, X):
        if not self.has_fit_been_run:
            raise ValueError("Model has not been fit! Run fit() before calling this.")
        reconstruction = self.deep_clusterer.predict(X)
        return reconstruction

    def save(self, path):
        self.deep_clusterer.save_weights(path)

    def load(self, path):
        self.deep_clusterer.load_weights(path)
        self.has_fit_been_run = True
        return self

    def error_measure(self, y_true, y_pred, axis = None):
        # y_true = y_true.astype(dtype=np.float64)
        # y_pred = y_pred.astype(dtype=np.float64)
        # return K.eval(K.sum(y_true * (K.log(y_true + K.epsilon()) - K.log(y_pred + K.epsilon())) - y_true + y_pred))
        # return np.sum(np.multiply(y_true, (np.log(y_true + K.epsilon()) - np.log(y_pred + K.epsilon())))
        #               - y_true + y_pred, axis=axis) \
        #        / float(y_true.shape[0])
        return np.mean(np.square(y_true - y_pred))

