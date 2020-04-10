import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime

import findMin


class softmaxClassifier:
    def __init__(self, lammy=0.01, maxEvals=100, alphaInit=1e-3, verbose=0):
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.alphaInit = alphaInit
        self.verbose = verbose

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes

        W = np.reshape(w, (k, d))

        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1

        XW = np.dot(X, W.T)
        Z = np.sum(np.exp(XW), axis=1)

        # Calculate the function value
        f = - np.sum(XW[y_binary] - np.log(Z)) + 0.5 * \
            self.lammy * norm(W, 'fro')**2
        g = (np.exp(XW) / Z[:, None] - y_binary).T@X + self.lammy * W
        return f, g.flatten()

    def fit(self, X, y):
        # adding bias
        b = np.ones((X.shape[0], 1))
        X = np.concatenate((X, b), axis=1)

        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(k*d)
        self.W, _ = findMin.findMin(self.funObj, self.W,
                                    self.maxEvals, X, y, verbose=self.verbose, alphaInit=self.alphaInit)
        self.W = np.reshape(self.W, (k, d))

    def predict(self, X):
        # adding bias
        b = np.ones((X.shape[0], 1))
        X = np.concatenate((X, b), axis=1)

        return np.argmax(X@self.W.T, axis=1)
