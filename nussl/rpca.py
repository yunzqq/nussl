#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal

class RPCA(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, high_pass_cutoff=None, num_iterations=None, epsilon=None,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(RPCA, self).__init__(input_audio_signal=input_audio_signal)
        self.high_pass_cutoff = 100.0 if high_pass_cutoff is None else float(high_pass_cutoff)
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft

        self.stft = None
        self.magnitude_spectrogram = None
        self.epsilon = 1e-7 if epsilon is None else epsilon
        self.num_iterations = 100 if num_iterations is None else num_iterations
        self.gain = 1
        self.error = None

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(self.high_pass_cutoff * (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate)) + 1

        self._compute_spectrum()

        # separate the mixture background by masking
        background_stft = []
        for i in range(self.audio_signal.num_channels):
            background_mask = self.compute_rpca_mask(self.magnitude_spectrogram[:, :, i])
            background_mask[0:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground

            # apply mask
            stft_with_mask = background_mask * self.stft[:, :, i]
            background_stft.append(stft_with_mask)

        background_stft = np.array(background_stft).transpose((1, 2, 0))
        self.background = AudioSignal(stft=background_stft, sample_rate=self.audio_signal.sample_rate)
        self.background.istft(self.stft_params.window_length, self.stft_params.hop_length,
                              self.stft_params.window_type, overwrite=True,
                              use_librosa=self.use_librosa_stft)

        return self.background
    
    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        self.magnitude_spectrogram = np.abs(self.stft)

    def compute_rpca_mask(self, magnitude_spectrogram):
        low_rank, sparse_matrix = self.decompose(magnitude_spectrogram)
        bg_mask = self.gain * np.abs(sparse_matrix) <=  np.abs(low_rank)
        return bg_mask
    
    def decompose(self, magnitude_spectrogram):
        #compute rule of thumb values of lagrange multiplier and svd-threshold
        _lambda = 1 / np.sqrt(np.max(magnitude_spectrogram.shape))
        
        #initialize low rank and sparse matrices to all zeros
        low_rank = np.zeros(magnitude_spectrogram.shape)
        sparse_matrix = np.zeros(magnitude_spectrogram.shape)
        
        #get singular values for magnitude_spectrogram
        two_norm = np.linalg.svd(magnitude_spectrogram, full_matrices = False, compute_uv = False)[0]
        inf_norm = np.linalg.norm(magnitude_spectrogram.flatten(), np.inf) / _lambda
        dual_norm = np.max([two_norm, inf_norm])
        residuals = magnitude_spectrogram / dual_norm

        #tunable parameters
        mu = 1.25 / two_norm
        mu_bar = mu * 1e7
        rho = 1.5

        error = np.inf
        converged = False
        num_iteration = 0
        
        while not converged and num_iteration < self.num_iterations:
            num_iteration += 1
            low_rank = self.svd_threshold(magnitude_spectrogram - sparse_matrix + residuals / mu, 1 / mu)
            sparse_matrix = self.shrink(magnitude_spectrogram - low_rank + residuals / mu, _lambda / mu)
            residuals += mu * (magnitude_spectrogram - low_rank - sparse_matrix)
            mu = np.min([mu * rho, mu_bar])
            error = np.linalg.norm(magnitude_spectrogram - low_rank - sparse_matrix, ord = 'fro') / np.linalg.norm(magnitude_spectrogram, ord = 'fro')
            if error < self.epsilon:
                converged = True
        self.error = error
        return low_rank, sparse_matrix

    def shrink(self, matrix, tau):
        return np.sign(matrix)*np.maximum(np.abs(matrix) - tau, 0)

    def svd_threshold(self, matrix, tau):
        u, sigma, v = np.linalg.svd(matrix, full_matrices = False)
        shrunk = self.shrink(sigma, tau)
        thresholded_singular_values = np.dot(u, np.dot(np.diag(shrunk), v))
        return thresholded_singular_values

    def reduced_rank_svd(self, matrix, k):
        u, sigma, v = np.linalg.svd(matrix, full_matrices = False)
        matrix_reduced = np.dot(u[:, 0:k], np.dot(sigma[0:K], v[0:k, :]))
        return matrix_reduced

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
        if self.background is None:
            return None

        self.foreground = self.audio_signal - self.background
        self.foreground.sample_rate = self.audio_signal.sample_rate
        return [self.background, self.foreground]