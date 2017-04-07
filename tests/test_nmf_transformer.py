import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import librosa

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal
from nussl.transformers import NMF
from nussl import Repet, Melodia, RepetSim, FT2D

signal = AudioSignal('../input/mixture/vocals.wav', duration = 30, offset=60)
signal.stft()
mag = signal.power_spectrogram_data

mag = librosa.logamplitude(mag, ref_power=np.max)[:, :, 0] + 80
plt.figure(figsize=(20, 8))
plt.subplot(311)
plt.imshow(mag, origin='lower', cmap='Greys', aspect='auto')
plt.xlim([0, mag.shape[1]])

model = NMF(n_components=100, alpha = .5, beta = .5)
model = model.fit(mag.T)

mixture_signal = AudioSignal('../input/mixture/mixture.wav', duration = 30, offset=60)
repet = FT2D(mixture_signal)
repet.run()
bg, fg = repet.make_audio_signals()
fg_mag = fg.power_spectrogram_data
fg_mag = librosa.logamplitude(fg_mag, ref_power=np.max)[:, :, 0] + 80

model.transform(fg_mag.T)
inverse_mag = model.inverse_transform(model.components_).T

plt.subplot(312)
plt.imshow(inverse_mag, origin='lower', cmap='Greys', aspect='auto')
error = model.reconstruction_error_by_frame(fg_mag.T)
print 'RepetSim', model.reconstruction_error(fg_mag.T)

plt.plot(error)
plt.xlim([0, mag.shape[1]])

energy = np.mean(fg_mag, axis=0)
error = np.multiply(error, energy)

repet = Melodia(mixture_signal)
repet.run()
bg, fg = repet.make_audio_signals()
fg_mag = fg.power_spectrogram_data
fg_mag = librosa.logamplitude(fg_mag, ref_power=np.max)[:, :, 0] + 80

model.transform(fg_mag.T)
inverse_mag = model.inverse_transform(model.components_).T

plt.subplot(313)
plt.imshow(inverse_mag, origin='lower', cmap='Greys', aspect='auto')


error = model.reconstruction_error_by_frame(fg_mag.T)
print 'Melodia', model.reconstruction_error(fg_mag.T)

energy = np.mean(fg_mag, axis=0)
error = np.multiply(error, energy)

plt.plot(error)
plt.xlim([0, mag.shape[1]])

plt.tight_layout()
plt.show()