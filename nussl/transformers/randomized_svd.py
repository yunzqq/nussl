from sklearn.base import TransformerMixin
import numpy as np
from scipy.linalg import orth
from sklearn.utils.extmath import safe_sparse_dot

class RandomizedSVD(TransformerMixin):
    def __init__(self, n_components = 150, compression_amount = .5, mode = 'compact'):
        self.n_components = n_components
        self.mode = mode
        self.compression_amount = compression_amount

    def fit_transform(self, X, y=None, **fit_params):
        # Step 1: generate n by 2K Gaussian iid matrix
        num_features, num_samples = X.shape
        K = self.n_components
        gamma = self.compression_amount

        X = X ** gamma

        gaussian_iid = np.random.randn(num_samples, np.min([2*K, num_samples]))

        # Step 2: form Y=A*gaussian_iid
        Y = np.dot(X, gaussian_iid)

        # Step 3: compute an orthonormal basis Q for the range of Y
        Q = orth(Y)

        # Step 4: form B=Q.T*X
        B = np.dot(np.conj(Q.T), X)

        # Step 5: compute svd of B
        U_est, S, V = np.linalg.svd(B, full_matrices=False)

        # Step 6: form U=Q*U)est
        U = np.dot(Q, U_est)

        # Step 7: update the # of components and matrix sizes
        K = np.min(np.array([K, np.shape(U)[1]]))
        U = U[:, 0:K]
        S = np.diag(S[0:K])
        V = V.T[:, 0:K]

        if self.mode == 'diagonal':
            S = np.diag(S)
        elif self.mode == 'compact':
            sqrtS = np.diag(np.sqrt(np.diag(S)))
            U = np.dot(U, sqrtS)
            V = np.dot(V, sqrtS)
            S = np.eye(K)

        self.row_vectors = U
        self.column_vectors = V
        self.singular_values = S

        return self

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        return safe_sparse_dot(self.row_vectors.T, X)

    def inverse_transform(self, X):
        return np.abs(np.dot(self.row_vectors, np.conj(self.column_vectors.T))) ** (1 / self.compression_amount)