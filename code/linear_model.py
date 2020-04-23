import numpy as np

import findMin

from sklearn.base import BaseEstimator, ClassifierMixin


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
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
            self.lammy * np.sum(W**2)
        g = (np.exp(XW) / Z[:, None] - y_binary).T@X + self.lammy * W
        return f, g.flatten()

    def fit(self, X, y):
        # adding bias
        b = np.ones((X.shape[0], 1))
        X = np.concatenate((X, b), axis=1)

        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k

        # Initial guess
        self.W = np.zeros(k*d)
        self.W, f = findMin.findMin(
            self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k, d))

    def predict(self, X):
        # adding bias
        b = np.ones((X.shape[0], 1))
        X = np.concatenate((X, b), axis=1)

        return np.argmax(X@self.W.T, axis=1)

    def predict_proba(self, X):
        # adding bias
        b = np.ones((X.shape[0], 1))
        X = np.concatenate((X, b), axis=1)

        return X@self.W.T


class MultiClassSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, lammy=0.01, epochs=100, learning_rate=1e-3, batch_size=200, verbose=0):
        self.lammy = lammy
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k, d))

        XW = X.dot(W.T)
        XW_correct_class = XW[np.arange(n), y]

        # Calculate the function value
        max_other_classes = np.maximum(
            0, 1-XW_correct_class[:, None] + XW)
        # adjust for the correct class
        max_other_classes[np.arange(n), y] = 0
        f = np.sum(max_other_classes)
        # Add L2 regularization
        f += 0.5 * self.lammy * np.sum(W**2)

        # Calculate the gradient value
        mask = (max_other_classes > 0).astype(int)
        incorrect_count = np.sum(mask, axis=1)
        mask[np.arange(n), y] = -incorrect_count
        g = mask.T.dot(X) + self.lammy * W

        return f, g.flatten()

    def fit(self, X, y):
        # use stochastic sub-gradient descent
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k

        # Initial guess
        self.W = np.zeros(k*d)

        # START SGDs
        for t in range(self.epochs):
            batch = np.random.choice(n, size=self.batch_size, replace=False)
            f, g = self.funObj(self.W, X[batch], y[batch])
            self.W -= self.learning_rate*g

        self.W = np.reshape(self.W, (k, d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

    def predict_proba(self, X):
        return X@self.W.T
