#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from librosa import display
from scipy.ndimage.filters import maximum_filter, minimum_filter

from .. import constants
import separation_base
from .. import AudioSignal
from keras.models import load_model

class AlgorithmSwitcher(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """

    def __init__(self, input_audio_signal, input_vocal_estimates, estimate_labels=None, model='vocal_sdr_predictor.model'):
        self.vocal_estimates = input_vocal_estimates
        if estimate_labels is None:
            self.estimate_labels = ['Estimate %d' for i in range(0, len(self.vocal_estimates))]
        else:
            self.estimate_labels = estimate_labels

        self.mixture = input_audio_signal
        self.model = load_model(model)
        self.background = None
        self.foreground = None
        super(AlgorithmSwitcher, self).__init__(input_audio_signal=input_audio_signal)


    def run(self):
        """

        Example:
             ::

        """
        resample_hz = 8000
        mixture_audio_data = self.mixture.resample(resample_hz).audio_data.reshape(-1, 2)
        mixture_audio_data = np.mean(mixture_audio_data, axis=-1)
        estimate_audio_data = []
        for vocal_estimate in self.vocal_estimates:
            estimate_audio_data.append(np.mean(vocal_estimate.resample(resample_hz).audio_data.reshape(-1, 2), axis = -1))
        estimate_audio_data = np.vstack(estimate_audio_data)

        mixture_audio_data = mixture_audio_data.reshape(8000, -1)
        estimate_audio_data = estimate_audio_data.reshape(8000, -1, len(self.vocal_estimates))
        self.sdrs = np.empty((mixture_audio_data.shape[-1], estimate_audio_data.shape[-1]))

        for sec in range(mixture_audio_data.shape[-1]):
            for est in range(estimate_audio_data.shape[-1]):
                input_data = np.vstack([mixture_audio_data[:, sec], estimate_audio_data[:, sec, est]])
                input_data = input_data.reshape((1,) + input_data.shape)

                sdr = self.model.predict(input_data)
                self.sdrs[sec, est] = sdr

        foreground_audio_data = np.zeros(self.mixture.audio_data.shape)
        estimate_audio_data = [x.audio_data for x in self.vocal_estimates]
        sample_rate = self.mixture.sample_rate
        best = np.argmax(self.sdrs, axis=1)

        for sec in range(mixture_audio_data.shape[-1]):
            foreground_audio_data[:, sec*sample_rate:(sec+1)*sample_rate] = \
                estimate_audio_data[best[sec]][:, sec*sample_rate:(sec+1)*sample_rate]

        self.foreground = AudioSignal(audio_data_array=foreground_audio_data, sample_rate=self.mixture.sample_rate)
        self.background = self.mixture - self.foreground

        return self.background, self.foreground

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

        # self.foreground = self.audio_signal - self.background
        # self.foreground.sample_rate = self.audio_signal.sample_rate
        return [self.background, self.foreground]

    def plot(self, output_name, **kwargs):
        fig = plt.gcf()
        cmap = plt.cm.get_cmap('jet')
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, len(self.estimate_labels), len(self.estimate_labels) + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        im = plt.imshow(np.vstack([np.argmax(self.sdrs, axis=1), np.argmin(self.sdrs, axis=1)]), aspect='auto', cmap=cmap, norm=norm)
        plt.yticks([0, 1], ['$Predicted$ $best$', '$Predicted$ $worst$'])
        plt.xlabel('Time (s)')

        cbar = fig.colorbar(im)
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(self.estimate_labels):
            cbar.ax.text(.5, (2 * j + 1) / (2.0 * len(self.estimate_labels)), lab, ha='center', va='center', size=18, color='white')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_aspect('auto')