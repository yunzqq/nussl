import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import librosa
import tensorflow as tf
from librosa.display import specshow


path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal
from nussl.transformers import RecurrentAutoEncoder
from nussl import Repet, Melodia, RepetSim, FT2D
signal = AudioSignal('../input/mixture/drums.wav', duration = 30, offset=30)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80

model = RecurrentAutoEncoder(input_shape=mag.shape[0], output_shape=mag.shape[0])

model = model.fit(mag.T, mag.T, epochs=300, batch_size=1000)
plt.figure(figsize=(20, 8))
plt.subplot(311)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

signal = AudioSignal('../input/mixture/drums.wav', duration = 30, offset=60)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80

plt.subplot(312)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

mixture_reconstruction = model.inverse_transform(mag.T).T

plt.subplot(313)
specshow(mixture_reconstruction, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

plt.tight_layout()
plt.show()

