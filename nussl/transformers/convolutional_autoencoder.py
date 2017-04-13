from keras.layers import Input, InputLayer, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, Lambda, Concatenate, Masking
from keras.layers.merge import Add
from keras.models import Model, Sequential, load_model
from keras import optimizers, objectives
from sklearn.base import TransformerMixin
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import operator
from deconvolutional import Conv2DTransposeSharedWeights


class ConvolutionalAutoEncoder(TransformerMixin):
    def __init__(self, input_shape=(25, 513), output_shape=(25, 513), activation_sparsity=0.01,
                 template_sparsity=1, num_decoders = 1, loss='mask_loss', optimizer='adadelta'):
        if loss == 'mask_loss':
            loss = self.mask_loss
        self.loss = loss
        self.optimizer = optimizer
        self.alpha = .001

        self.num_timesteps, self.num_features = input_shape
        num_frequency_filters = 50
        num_time_filters = 30
        bottleneck_size = 128
        kernel_size_vertical = (1, self.num_features)
        kernel_size_horizontal = (3, 1)

        self.input_frame = Input(shape=input_shape)
        encoder = Sequential()

        #encoding
        #encoder.add(InputLayer(input_shape=input_shape)(input_frame))
        #noise_layer = GaussianNoise(.1, input_shape=(input_shape,))
        #network.add(noise_layer)
        vertical_convolution = Conv2D(num_frequency_filters, kernel_size_vertical, strides=(1, 1), padding='valid', activation='linear')
        #vertical_bias = Dense(50, kernel_initializer='ones', use_bias=True)
        encoder.add(Reshape((self.num_timesteps, self.num_features, -1), input_shape=input_shape))
        encoder.add(vertical_convolution)
        horizontal_convolution = Conv2D(num_time_filters, kernel_size_horizontal, strides=(1, 1), padding='valid', activation='linear')
        # horizontal_bias = Dense(30, kernel_initializer='zeros')
        encoder.add(horizontal_convolution)
        encoder.add(Flatten())
        encoder.add(Dense(bottleneck_size, activation='relu'))

        #decoding
        def create_decoder_for_source(decoder):
            decoder.add(Dense(reduce(operator.mul, list(horizontal_convolution.output_shape[1:])), activation = 'relu',
                              input_shape = (bottleneck_size,)))
            decoder.add(Reshape(list(horizontal_convolution.output_shape[1:])))
            decoder.add(Conv2DTransposeSharedWeights(num_frequency_filters, kernel_size_horizontal,
                                        shared_layer=horizontal_convolution, padding='valid', activation='linear', use_bias=False))

            decoder.add(Conv2DTransposeSharedWeights(1, kernel_size_vertical,
                                        shared_layer=vertical_convolution, padding='valid', activation='linear', use_bias=False))
            decoder.add(Reshape(input_shape))
            return decoder

        #one decoder per output source
        self.decoders = []
        for num_decoder in range(num_decoders):
            self.decoders.append(create_decoder_for_source(Sequential()))

        #connect the encoder and decoder modules to the input
        encoder = encoder(self.input_frame)
        self.decoders = [decoder(encoder) for decoder in self.decoders]
        self.autoencoder = Model(inputs=[self.input_frame], outputs=self.decoders)
        self.has_fit_been_run = False
        self.autoencoder.compile(loss=self.loss, optimizer=self.optimizer)

    def mask_loss(self, y_true, y_pred):
        all_sources = K.concatenate(self.decoders, axis=1)
        all_sources = K.sum(K.reshape(all_sources, (-1, len(self.decoders), self.num_timesteps, self.num_features)), axis=1)
        interfering_source_mask = ((all_sources - y_pred) + K.epsilon()) / (all_sources + K.epsilon())
        mask = (y_pred + K.epsilon()) / (all_sources + K.epsilon())
        target_source = mask * self.input_frame
        interfering_sources = interfering_source_mask * self.input_frame
        contrastive_mask_loss = objectives.mean_squared_error(target_source, interfering_sources)
        target_mask_loss = objectives.mean_squared_error(y_true, target_source)
        return target_mask_loss - self.alpha*contrastive_mask_loss

    def fit(self, input_data, output_data, **kwargs):
        self.autoencoder.fit(input_data, output_data,
                             **kwargs)
        self.has_fit_been_run = True
        return self

    def fit_generator(self, *args, **kwargs):
        self.autoencoder.fit_generator(*args, **kwargs)
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
        reconstruction = self.autoencoder.predict(X)
        loss = self.error_measure(X, reconstruction[0]) - self.error_measure(X, reconstruction[1])
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
        self.autoencoder.save_weights(path)

    def load(self, path):
        self.autoencoder.load_weights(path)
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

