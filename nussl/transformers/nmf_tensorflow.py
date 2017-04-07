from sklearn.base import TransformerMixin
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF

class NMF(TransformerMixin):
    def __init__(self, n_components = 20, num_iterations = 1000, alpha = .9, beta = .9):
        self.num_iterations = num_iterations
        self.num_components = n_components
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y=None, **fit_params):
        tf.reset_default_graph()
        with tf.Session() as sess:
            tfnmf = TFNMF(X, self.num_components, algo='sparse_mu', alpha = self.alpha, beta = self.beta)
            self.components_, self.activations_ = tfnmf.run(sess, max_iter=self.num_iterations)
            self.components_ = np.array(self.components_)
            self.activations_ = np.array(self.activations_)
        return self

    def transform(self, X):
        tf.reset_default_graph()
        with tf.Session() as sess:
            tfnmf = TFNMF(X, self.num_components, update_H = False, H = self.activations_, algo='sparse_mu')
            self.components_, self.activations_ = tfnmf.run(sess, max_iter=self.num_iterations)
            self.components_ = np.array(self.components_)
            self.activations_ = np.array(self.activations_)
        return self.components_

    def reconstruction_error(self, X):
        reconstruction = self.inverse_transform(self.transform(X))
        return np.linalg.norm(X - reconstruction, 'fro')

    def reconstruction_error_by_frame(self, X):
        reconstruction = self.inverse_transform(self.transform(X))
        return np.linalg.norm(X - reconstruction, axis=1)

    def inverse_transform(self, X):
        return np.dot(X, self.activations_)