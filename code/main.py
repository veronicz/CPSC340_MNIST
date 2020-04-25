import argparse
import gzip
import os
import pickle
import time

import numpy as np
from numpy.linalg import norm
from scipy.ndimage import interpolation
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

from cnn import CNN
from deskewing import deskewMNIST
from knn import KNN
from linear_model import MultiClassSVM, SoftmaxClassifier
from optimizer import optimizeCNN, optimizeLR, optimizeMLP, optimizeSVM


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


def loadDeskewedMNIST():
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set

    try:
        X_train_deskewed = np.load("X_train_deskewed.npy")
        X_test_deskewed = np.load("X_test_deskewed.npy")
    except IOError:
        deskewMNIST(X, Xtest)
        X_train_deskewed = np.load("X_train_deskewed.npy")
        X_test_deskewed = np.load("X_test_deskewed.npy")

    return X_train_deskewed, y, X_test_deskewed, ytest


def cross_validate_error(model, X, y, n_folds):
    v_error = []
    for train_index, test_index in KFold(n_folds).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = np.mean(y_pred != y_test)
        v_error.append(error)

    return np.mean(v_error)


def optimizeKNNHyper(verbose=0):
    min_err = 1
    k_opt = 1
    k_list = range(1, 30)
    val_err_list = [1]
    for k in k_list:
        val_err = cross_validate_error(KNN(k=k), X_train_deskewed, y, 5)
        if verbose > 0:
            print(
                f"Error for k={k}: ", val_err)
        if val_err <= min_err:  # choose the simper model if error is the same
            min_err = val_err
            k_opt = k

        val_err_list.append(val_err)
        last_val_errs = val_err_list[-5:]
        # stop if the validation error has been increasing for the last 5 k's
        if last_val_errs == sorted(last_val_errs):
            if verbose > 0:
                print("Pruning because validation error is increasing...")
            break  # model is becoming too simple

        if verbose > 0:
            print("Best k selected: ", k_opt)
    return k_opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "KNN":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()

        t = time.time()
        k_opt = optimizeKNNHyper()
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model = KNN(k=k_opt)
        model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print("KNN test error for k=%d: %.5f" % (k_opt, test_err))

    elif question == "LR":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()

        t = time.time()
        model, hyperparams = optimizeLR(X_train_deskewed, y, 5, verbose=1)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(f"Softmax test error for {hyperparams}: %.5f" % test_err)

    elif question == "SVM":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()

        t = time.time()
        model, hyperparams = optimizeSVM(X_train_deskewed, y, 5, verbose=1)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(f"SVM test error for {hyperparams}: %.5f" % test_err)

    elif question == "MLP":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()
        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        t = time.time()
        model, params = optimizeMLP(X_train_deskewed, Y, 5, verbose=1)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model.fit(X_train_deskewed, Y)
        print("Fitting took %d seconds" % (time.time()-t))

        # Compute test error
        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(f"MLP test error for {params}= ", test_err)

    elif question == "CNN":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()
        y = y.reshape(y.shape[0], 1)

        t = time.time()
        model, hyperparams = optimizeCNN(X_train_deskewed, y, 2, verbose=1)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        cnn_params = model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(f"CNN test error for {hyperparams}:  %.5f" % (test_err))

    else:
        print("Unknown question: %s" % question)
