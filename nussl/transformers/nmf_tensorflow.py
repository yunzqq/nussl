from sklearn.base import TransformerMixin
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF

class NMF(TransformerMixin):
    def __init__(self, n_components = 20, num_iterations = 20):
        self.num_iterations = num_iterations
        self.num_components = n_components

    def fit(self, X, y=None, **fit_params):
        tf.reset_default_graph()
        with tf.Session() as sess:

            tfnmf = TFNMF(X, self.num_components)
            self.components_, self.activations_ = tfnmf.run(sess, max_iter=self.num_iterations)
            self.components_ = np.mat(self.components_)
            self.activations_ = np.mat(self.activations_)
        return self

    def transform(self, X):
        tf.reset_default_graph()
        with tf.Session() as sess:

            tfnmf = TFNMF(X, self.num_components, update_H = False, H = self.activations_)
            tfnmf._build_mu_algorithm()
            self.components_, self.activations_ = tfnmf.run(sess, max_iter=self.num_iterations)
            self.components_ = np.mat(self.components_)
        return self.components_

    def inverse_transform(self, X):
        return np.dot(X, self.activations_)