#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projet for multicue separation
from this paper:

@inproceedings{fitzgeraldPROJETa,
TITLE = {{PROJET - Spatial Audio Separation Using Projections}},
AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
ADDRESS = {Shanghai, China},
PUBLISHER = {{IEEE}},
YEAR = {2016},
}

Copyright (c) 2016, Antoine Liutkus, Inria

modified by Ethan Manilow and Prem Seetharaman for incorporation into nussl.
"""

import numpy as np

import separation_base
import constants
from audio_signal import AudioSignal
from transformers import RandomizedSVD, NMF
from ft2d import FT2D

import matplotlib.pyplot as plt

class ProjetRepet(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, use_librosa_stft=constants.USE_LIBROSA_STFT, num_sources=None,
                 num_iterations=None, num_panning_directions=None, num_projections=None, verbose=None,
                 num_components=None):
        super(ProjetRepet, self).__init__(input_audio_signal=input_audio_signal)
        self.sources = None
        self.use_librosa_stft = use_librosa_stft
        self.stft = None
        self.num_sources = 6 if num_sources is None else num_sources
        self.num_iterations = 50 if num_iterations is None else num_iterations
        self.num_panning_directions = 41 if num_panning_directions is None else num_panning_directions
        self.num_projections = 15 if num_projections is None else num_projections
        self.verbose = False if verbose is None else verbose
        self.num_components = 400 if num_components is None else num_components
        if self.audio_signal.num_channels == 1:
            raise ValueError('Cannot run PROJET on a mono audio signal!')

        self.P = None
        self.Q = None


    def run(self):
        """

        Returns:
            sources (list of AudioSignals): A list of AudioSignal objects with all of the sources found in the mixture

        Example:
             ::

        """
        self._compute_spectrum()

        (F, T, I) = self.stft.shape
        repet = FT2D(self.audio_signal)
        repet.run()
        bg, fg = repet.make_audio_signals()
        foreground_spectrogram = np.abs(fg.stft()) ** 2
        foreground_spectrogram = np.mean(foreground_spectrogram, axis=-1)
        model = NMF(n_components=self.num_components)
        print 'Fitting model to foreground'
        model.fit(foreground_spectrogram.T)


        num_sources = self.num_sources
        num_possible_panning_directions = self.num_panning_directions
        num_projections = self.num_projections
        eps = 1e-20
        # initialize PSD and panning to random
        # P is size of flattened stft by number of sources (e.g. (F, T, 4)
        self.P = np.abs(np.random.randn(self.num_components*T, num_sources), dtype='float32') + 1
        # Q is number of panning directions to look for by number of sources (e.g. 41 x 4)
        self.Q = np.abs(np.random.randn(num_possible_panning_directions, num_sources), dtype='float32') + 1


        # compute panning profiles
        # 30 for regular grid, the others as random
        # default (2, 41 - 30) concatenated with (2, 30) -> (2, 41)
        panning_matrix = np.concatenate((self.complex_randn((I, num_possible_panning_directions - 30)),
                                         self.multichannelGrid(I, 30)), axis=1)
        panning_matrix /= np.sqrt(np.sum(np.abs(panning_matrix) ** 2, axis=0))[None, ...]

        # compute projection matrix
        # 5 for orthogonal to a regular grid, the others as random
        # (15 - 5, 2) + (2, 5).T -> (5, 2) -> (15, 2)
        projection_matrix = np.concatenate((self.complex_randn((max(num_projections - 5, 0), I)),
                                            self.orthMatrix(self.multichannelGrid(I, min(num_projections, 5)))))
        projection_matrix /= np.sqrt(np.sum(np.abs(projection_matrix) ** 2, axis=1))[..., None]

        # compute K matrix
        # (15, 2) x (2, 41) = (15, 41).
        K = np.abs(np.dot(projection_matrix, panning_matrix)).astype(np.float32)

        # compute the projections and store their spectrograms and squared spectrograms
        # (F, T, 2) (15, 2). first by 2,1 axes -> (F, T, 2) x (2, 15) -> (F, T, 15). and then flatten it.
        C = np.tensordot(self.stft, projection_matrix, axes=(2, 1))


        # NOTE: C now the same shape as P.

        print 'Compressing projections'
        V = np.abs(C).astype(np.float32)
        V2 = V ** 2
        V2 = np.reshape(V2, (F, T * num_projections))
        V2 = model.transform(V2.T).T

        # compressed = np.empty((self.num_components, T, num_projections))
        # for num_projection in range(num_projections):
        #     print 'Projection %d' % num_projection
        #     compressed[:, :, num_projection] = model.transform(V2[:, :, num_projection].T).T
        # V2 = compressed

        print V2.shape, (self.num_components, T, num_projections)
        V2 = np.reshape(V2, (self.num_components * T, num_projections))
        C = []  # release memory

        # main iterations
        for iteration in range(self.num_iterations):
            if self.verbose:
                print 'Iteration %d' % iteration
            # np.dot(Q.T, K.T) -> (e.g. (4, 41) by (41, 15) -> (4, 15)
            # (F*T, 4) by (4, 15) -> (F*T, 15).
            sigma = np.dot(self.P, np.dot(self.Q.T, K.T))
            # np.dot(K, Q) -> (15, 41) x (41, 4) -> (15, 4)
            # (F*T, 15), (15, 4) -> (F*T, 4) / (F*T, 4)
            # updating P
            self.P *= np.dot(1.0 / (sigma + eps), np.dot(K, self.Q)) / \
                      (np.dot(3 * sigma / (sigma ** 2 + V2 + eps), np.dot(K, self.Q)))
            # the following line is an optional trick that enforces orthogonality of the spectrograms.
            # self.P*=(100+self.P)/(100+np.sum(self.P,axis=1)[...,None])
            # update sigma using updated P. transpose to fit into Q. (15, F*T)
            sigma = np.dot(self.P, np.dot(self.Q.T, K.T)).T
            # updating Q
            # (41, 15) dot ((15, F*T) dot (F*T, 4) - > (15, 4) dot ((41, 15) dot (15, 4) -> (41, 4))
            # (41, 4) dot (41, 15), [(15, F*T), (F*T, 4)] -> (15, 4)
            # (41, 4)

            self.Q *= np.dot(K.T, np.dot(1.0 / (sigma + eps), self.P)) / \
                      np.dot(K.T, np.dot(3 * sigma / (sigma ** 2 + V2.T + eps), self.P))


        # final separation
        # 2 by 15
        recompose_matrix = np.linalg.pinv(projection_matrix)  # IxM
        # expand back out P
        print 'Expanding P'
        print self.P.shape

        self.P = np.reshape(self.P, (self.num_components, T*num_sources))
        self.P = model.inverse_transform(self.P.T).T
        self.P = np.reshape(self.P, (F * T, num_sources))

        # final sigma, is (F*T, 15)
        sigma = np.dot(self.P, np.dot(self.Q.T, K.T))
        # project mixtures again? (F*T, 15)
        C = np.dot(np.reshape(self.stft, (F * T, I)), projection_matrix.T)

        self.sources = []

        for j in range(num_sources):
            sigma_j = np.outer(self.P[:, j], np.dot(self.Q[:, j].T, K.T))
            source_stft = sigma_j / sigma * C
            source_stft = np.dot(source_stft, recompose_matrix.T)
            source_stft = np.reshape(source_stft, (F, T, I))
            source = AudioSignal(stft = source_stft, sample_rate = self.audio_signal.sample_rate)
            source.istft(self.stft_params.window_length, self.stft_params.hop_length,
                        self.stft_params.window_type, overwrite=True,
                        use_librosa=self.use_librosa_stft,
                        truncate_to_length=self.audio_signal.signal_length)
            self.sources.append(source)

        return self.sources

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

    def multichannelGrid(self, I, L, sigma=1, normalize=True):
        # 15 points equally spaced between 0 and num_channels - 1 (1). 15 points between 0 and 1 basically.
        pos = np.linspace(0, I - 1, L)
        # 2 by 15 all 0s.
        res = np.zeros((I, L))
        for i in range(I):
            # each row becomes e^(+/-[0, 1]**2)
            res[i, ...] = np.exp(-(pos - i) ** 2 / sigma ** 2)
        if normalize:
            res /= np.sqrt(np.sum(res ** 2, axis=0))
        return res

    def complex_randn(self, shape):
        #return np.ones(shape) + 1j * np.zeros(shape)
        return np.random.randn(*shape) + 1j * np.random.randn(*shape)


    def orthMatrix(self, R):
        # 2 by 15
        (I, L) = R.shape
        # 15 by 2
        res = np.ones((L, I))
        # all rows, squeeze removes all one dimensional entries (1, 3, 1) shape goes to (3) shape.
        # sum of all rows of R but the last one, along each column.
        # all columns of res but the last one become the last row of R (2 by 1)
        # divided by the sum of all the columns of R but the last one.
        # transpose to fit into res.
        res[:, -1] = - (R[-1, :] / np.squeeze(np.sum(R[:-1, :], axis=0))).T
        # normalize res by rms along each row
        res /= np.sqrt(np.sum(res ** 2, axis=1))[..., None]
        return res

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run FT2D.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
        """
        if self.sources is None:
            return None

        return self.sources
