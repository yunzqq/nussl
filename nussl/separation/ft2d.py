#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import display
from matplotlib import gridspec
from scipy.ndimage.filters import maximum_filter, minimum_filter

from .. import constants
import separation_base
from nussl.audio_signal import AudioSignal


class FT2D(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, high_pass_cutoff=None, footprint=None,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(FT2D, self).__init__(input_audio_signal=input_audio_signal)
        self.high_pass_cutoff = 100.0 if high_pass_cutoff is None else float(high_pass_cutoff)
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft
        self.footprint = np.ones((1, 25)) if footprint is None else footprint

        self.stft = None
        self.ft2d = None
        self.bg_ft2d = []
        self.fg_ft2d = []
        self.bg_inversion = []
        self.fg_inversion = []

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

        # the MATLAB implementation had
        self._compute_spectrum()

        # separate the mixture background by masking
        background_stft = []
        foreground_stft = []
        for i in range(self.audio_signal.num_channels):
            background_mask = self.compute_ft2d_mask(self.ft2d[:, :, i])
            background_mask[0:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground

            # apply mask

            stft_with_mask = background_mask * self.stft[:, :, i]
            background_stft.append(stft_with_mask)
            stft_with_mask = (1 - background_mask) * self.stft[:, :, i]
            foreground_stft.append(stft_with_mask)

        background_stft = np.array(background_stft).transpose((1, 2, 0))
        self.background = AudioSignal(stft=background_stft, stft_params = self.stft_params, sample_rate=self.audio_signal.sample_rate)
        self.background.istft(overwrite=True, use_librosa=self.use_librosa_stft, truncate_to_length=self.audio_signal.signal_length)

        foreground_stft = np.array(foreground_stft).transpose((1, 2, 0))
        self.foreground = AudioSignal(stft=foreground_stft, stft_params=self.stft_params, sample_rate=self.audio_signal.sample_rate)
        self.foreground.istft(overwrite=True, use_librosa=self.use_librosa_stft, truncate_to_length=self.audio_signal.signal_length)
        
        return self.background, self.foreground

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        self.ft2d = np.stack([np.fft.fft2(np.abs(self.stft[:, :, i]))
                              for i in range(self.audio_signal.num_channels)], axis = -1)

    def compute_ft2d_mask(self, ft2d):
        bg_ft2d, fg_ft2d = self.filter_local_maxima(ft2d)
        bg_stft = np.fft.ifft2(bg_ft2d)
        fg_stft = np.fft.ifft2(fg_ft2d)
        self.bg_inversion.append(bg_stft)
        self.fg_inversion.append(fg_stft)
        self.bg_ft2d.append(bg_ft2d)
        self.fg_ft2d.append(fg_ft2d)
        bg_mask = bg_stft > fg_stft
        #smoothing out the mask - maybe not helpful
        #kernel =  np.full((1, 5), 1/5.)
        #bg_mask = convolve(bg_mask, kernel)
        return bg_mask

    def filter_local_maxima(self, ft2d):
        data = np.abs(np.fft.fftshift(ft2d))
        data = data / np.max(data)
        threshold = np.std(data)
        
        data_max = maximum_filter(data, footprint = self.footprint)
        maxima = (data == data_max)
        data_min = minimum_filter(data, footprint = self.footprint)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        maxima = np.maximum(maxima, np.flipud(np.fliplr(maxima)))
        #maxima[0:maxima.shape[0]/2, 0:maxima.shape[1]/2] = 0
        #maxima[maxima.shape[0] / 2:, maxima.shape[1] / 2:] = 0
        maxima = np.fft.ifftshift(maxima)
        
        background_ft2d = np.multiply(maxima, ft2d)
        foreground_ft2d = np.multiply(1 - maxima, ft2d)
        return background_ft2d, foreground_ft2d

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

        #self.foreground = self.audio_signal - self.background
        #self.foreground.sample_rate = self.audio_signal.sample_rate
        return [self.background, self.foreground]

    def plot(self, output_name, **kwargs):
        fig = plt.gcf()

        def plot_image(fig, location, matrix, title, x=None, y=None):
            fig.add_subplot(location)
            plt.title(title)
            display.specshow(matrix, x_axis=x, y_axis=y, hop_length=.5*self.stft_params.hop_length)

        gs = gridspec.GridSpec(6, 10)
        for i in range(self.audio_signal.num_channels):
            plot_image(fig, gs[0:2, 0:7], librosa.amplitude_to_db(np.abs(self.stft[:, :, i]), ref=np.max), 'Mixture STFT', 'time', 'log')
            plot_image(fig, gs[0:2, 7:], librosa.amplitude_to_db(np.abs(np.fft.fftshift(self.ft2d[:, :, i]))), 'Mixture 2DFT')

            plot_image(fig, gs[2:4, 0:7], librosa.amplitude_to_db(self.bg_inversion[i]), 'STFT from inverted background 2DFT', 'time',
                 'log')
            plot_image(fig, gs[2:4, 7:], librosa.amplitude_to_db(np.abs(np.fft.fftshift(self.bg_ft2d[i]))), 'Background 2DFT')

            #plot_image(fig, gs[2, 0:7], librosa.amplitude_to_db(np.abs(self.background.stft()[:, :, i])), 'Background STFT', 'time', 'log')

            plot_image(fig, gs[4:, 0:7], librosa.amplitude_to_db(np.abs(self.fg_inversion[i])), 'STFT from inverted foreground 2DFT',
                 'time', 'log')
            plot_image(fig, gs[4:, 7:], librosa.amplitude_to_db(np.abs(np.fft.fftshift(self.fg_ft2d[i]))), 'Foreground 2DFT')

            #plot_image(fig, gs[4, 0:7], librosa.amplitude_to_db(np.abs(self.foreground.stft()[:, :, i])), 'Foreground STFT', 'time', 'log')
            break
