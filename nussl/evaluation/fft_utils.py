from .. import constants
import scipy.fftpack
import numpy as np

if not constants.USE_GPU:
    fft = scipy.fftpack.fft
    ifft = scipy.fftpack.ifft
else:
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .1

    sess = tf.Session(config=config)
    input = tf.placeholder(tf.float32)
    fft_node = tf.fft(tf.complex(input, tf.zeros(tf.shape(input))), name='fft')
    ifft_node = tf.ifft(tf.complex(input, tf.zeros(tf.shape(input))), name='ifft')

    def fft(x, n=None, axis=-1):
        transformed = sess.run(fft_node, {input: x})
        return np.array(transformed)

    def ifft(x, n=None, axis=-1):
        transformed = sess.run(ifft_node, {input: x})
        return np.real(np.array(transformed))