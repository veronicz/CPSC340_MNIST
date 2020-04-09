import math

import numpy as np
import scipy as sp
from scipy.ndimage import interpolation


def deskewMNIST(X, Xtest):
    X_train_deskewed = deskewAll(X)
    X_test_deskewed = deskewAll(Xtest)
    np.save("X_train_deskewed", X_train_deskewed)
    np.save("X_test_deskewed", X_test_deskewed)


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28, 28)).flatten())
    return np.array(currents)


def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1]/v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


def moments(image):
    # A trick in numPy to create a mesh grid
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0*image)/totalImage  # mu_x
    m1 = np.sum(c1*image)/totalImage  # mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage  # var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage  # var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage  # covariance(x,y)
    # Notice that these are \mu_x, \mu_y respectively
    mu_vector = np.array([m0, m1])
    # Do you see a similarity between the covariance matrix
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix

