import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import approx_fprime
from scipy.spatial import distance


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y) == 0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Xtest ** 2, axis=1)[None]
        - 2 * np.dot(X, Xtest.T)
    )

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w, X, y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
                        (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def initializeFilter(size, scale=1.0):
    '''
    Initialize filter using a normal distribution with and a
    standard deviation inversely proportional the square root of the number of units
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    '''
    Initialize weights with a random normal distribution
    '''
    return np.random.standard_normal(size=size) * 0.01


def categoricalCrossEntropy(probs, label):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs))
