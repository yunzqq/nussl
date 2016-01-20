# In this demo the original REPET algorithm is tested

from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt

plt.interactive('True')
import numpy as np
from nussl.FftUtils import f_stft
from nussl.REPET_org import repet
import time

# close all figure windows
plt.close('all')

raise DeprecationWarning('Don\'t get used to using this. It\'s going away soon!')

# load the audio file
# fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/Sample1.wav')
fs, x = read('../Input/mix4.wav')

# scale to -1.0 to 1.0
convert_16_bit = float(2 ** 15)
x = x / (convert_16_bit + 1.0)

x = np.mat(x)
t = np.mat(np.arange(np.shape(x)[1]) / float(fs))

# generate and plot the spectrogram of the mixture   
L = 2048
win = 'Hamming'
ovp = 0.5 * L
nfft = L
mkplot = 1
fmax = 5000

# plt.figure(1)
# plt.title('Mixture')
Sm = f_stft(np.mat(x), num_ffts=nfft, win_length=L, window_type=win, window_overlap=ovp, sample_rate=fs)

# separation
start_time = time.clock()
y_org = repet(np.mat(x), fs)
print time.clock() - start_time, "seconds"

# plot the background and foreground
# plt.figure(3)
# plt.subplot(2, 1, 1)
# plt.title('Background time-domain signal')
# plt.plot(t.T, y_org.T)
# plt.axis('tight')
# plt.show()
# plt.subplot(2, 1, 2)
# plt.title('Foreground time-domain signal')
# plt.plot(t.T, (x - y_org).T)
# plt.axis('tight')
# plt.show()

# plt.figure(4)
# plt.subplot(2, 1, 1)
# plt.title('Background Spectrogram')
Sb = f_stft(np.mat(y_org), num_ffts=nfft, win_length=L, window_type=win, window_overlap=ovp, sample_rate=fs)
# plt.show()
# plt.subplot(2, 1, 2)
# plt.title('Foreground Spectrogram')
Sf = f_stft(np.mat(x - y_org), num_ffts=nfft, win_length=L, window_type=win, window_overlap=ovp, sample_rate=fs)
# plt.show()

# check whether the separated spectrograms add up to the original spectrogram
Spec_diff = np.abs(Sm[0] - (Sb[0] + Sf[0]))

if Spec_diff.max() < 1e-10:
    print('Background and foreground add up to the original mixture.')

# record the separated background and foreground in .wav files
filePath = '../Output/'
write(filePath + 'repetOrgBackground.wav', fs, y_org.T)
write(filePath + 'repetOrgForeground.wav', fs, (x - y_org).T)