import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import librosa
from librosa.display import specshow


path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal
from nussl.transformers import BasicAutoEncoder

signal = AudioSignal('../input/mixture/vocals.wav', duration = 30, offset=30)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80
mag = mag / np.max(mag)

model = BasicAutoEncoder(encoding_dim=512, input_shape=mag.shape[0], output_shape=mag.shape[0])

model = model.fit(mag.T, mag.T, epochs=300, batch_size=300)
plt.figure(figsize=(20, 8))
plt.subplot(311)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

model.save('model.h5')
print 'saved model'
del model
model = BasicAutoEncoder().load('model.h5')
print 'loaded model'

signal = AudioSignal('../input/mixture/vocals.wav', duration = 30, offset=30)
signal.stft()
mag = signal.power_spectrogram_data[:, :, 0]
mag = librosa.logamplitude(mag, ref_power=np.max) + 80
mag = mag / np.max(mag)

plt.subplot(312)
specshow(mag, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

mixture_reconstruction = model.inverse_transform(mag.T).T

plt.subplot(313)
specshow(mixture_reconstruction, y_axis='cqt', x_axis='time', sr=signal.sample_rate)

plt.tight_layout()
plt.show()