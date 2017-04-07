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
from nussl.transformers import ContextDeepAutoEncoder

def add_context(input_data, context):
    left = right = (context - 1) / 2
    tuples = [range(i - left, i + right + 1) for i in range(mag.shape[1])]
    tuples = tuples[left:-right]
    mag_input = np.transpose(mag[:, np.array(tuples)], (2, 0, 1))
    mag_input = mag_input.reshape((mag_input.shape[0] * mag_input.shape[1], mag_input.shape[2]))
    mag_input += np.random.random(mag_input.shape) * .1
    return mag_input, left

signal = AudioSignal('../input/mixture/vocals.wav', duration=30, offset=30)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80
mag = mag / np.max(mag)

context = 11
model = ContextDeepAutoEncoder(input_shape=mag.shape[0]*context, encoding_dim=150)
mag_input, num_frames = add_context(mag, context)

model = model.fit(mag_input.T, mag.T[num_frames:-num_frames], epochs=300, batch_size=1000)
plt.figure(figsize=(20, 8))
plt.subplot(311)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

signal = AudioSignal('../input/mixture/mixture.wav', duration = 30, offset=90)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80
mag = mag / np.max(mag)

mag_input, _ = add_context(mag, context)

plt.subplot(312)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

mixture_reconstruction = model.inverse_transform(mag_input.T).T

mask = (mixture_reconstruction**2)
mask = mask / np.max(mask)
stft = signal.stft()[:, num_frames:-num_frames, :]
separated = np.reshape(np.hstack([mask, mask]), stft.shape) * stft
signal = AudioSignal(stft=separated)
signal.istft()
signal.write_audio_to_file('separated.wav')

separated = np.reshape(np.hstack([1-mask, 1-mask]), stft.shape) * stft
signal = AudioSignal(stft=separated)
signal.istft()
signal.write_audio_to_file('residual.wav')

plt.subplot(313)
specshow(mixture_reconstruction, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

plt.tight_layout()
plt.show()