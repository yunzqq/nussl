from sklearn.base import TransformerMixin
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

class RobustPCA(TransformerMixin):
    def __init__(self, num_iterations = 20):
        self.num_iterations = num_iterations

    def fit_transform(self, X, y=None, **fit_params):


        return self

    def fit(self, X):
        self.fit_transform(X)
        return self

    def inverse_transform(self, X):
        return

    def fit_transform(self, X):
        # compute rule of thumb values of lagrange multiplier and svd-threshold
        _lambda = 1 / np.sqrt(np.max(X.shape))

        # initialize low rank and sparse matrices to all zeros
        self.low_rank = np.zeros(X.shape)
        self.sparse_matrix = np.zeros(X.shape)

        # get singular values for magnitude_spectrogram
        two_norm = np.linalg.svd(X, full_matrices=False, compute_uv=False)[0]
        inf_norm = np.linalg.norm(X.flatten(), np.inf) / _lambda
        dual_norm = np.max([two_norm, inf_norm])
        residuals = X / dual_norm

        # tunable parameters
        mu = 1.25 / two_norm
        mu_bar = mu * 1e7
        rho = 1.5

        error = np.inf
        converged = False
        num_iteration = 0

        while not converged and num_iteration < self.num_iterations:
            if self.verbose:
                print num_iteration, error
            num_iteration += 1
            low_rank = self.svd_threshold(X - sparse_matrix + residuals / mu, 1 / mu)
            sparse_matrix = self.shrink(X - low_rank + residuals / mu, _lambda / mu)
            residuals += mu * (X - low_rank - sparse_matrix)
            mu = np.min([mu * rho, mu_bar])
            error = np.linalg.norm(X - low_rank - sparse_matrix, ord='fro') / np.linalg.norm(
                X, ord='fro')
            if error < self.epsilon:
                converged = True
        self.error = error
        return low_rank, sparse_matrix

    def shrink(self, matrix, tau):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0)

    def svd_threshold(self, matrix, tau):
        u, sigma, v = np.linalg.svd(matrix, full_matrices=False)
        shrunk = self.shrink(sigma, tau)
        thresholded_singular_values = np.dot(u, np.dot(np.diag(shrunk), v))
        return thresholded_singular_values

    def reduced_rank_svd(self, matrix, k):
        u, sigma, v = np.linalg.svd(matrix, full_matrices=False)
        matrix_reduced = np.dot(u[:, 0:k], np.dot(sigma[0:K], v[0:k, :]))
        return matrix_reduced
