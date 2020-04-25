import numpy as np
from scipy import stats
import utils
from heapq import nlargest


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        T = Xtest.shape[0]
        y_pred = np.zeros(T)
        X = self.X
        y = self.y
        k = min(self.k, X.shape[0])
        distances = utils.euclidean_dist_squared(X, Xtest)
        # early stop when k smallest found
        knn_ind = np.argpartition(distances, k, axis=0)[:k]

        for t in range(T):
            y_pred[t] = utils.mode(y[knn_ind[:, t]])

        return y_pred
