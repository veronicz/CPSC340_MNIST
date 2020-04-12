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

from deskewing import deskewMNIST
from knn import KNN
from linear_model import MultiClassSVM, SoftmaxClassifier


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


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


def optimizeLinearClassifierHyper(modelConstructor, X, y, verbose=0):
    lambda_list = [5, 1, 0.01, 1e-3]
    alpha_list = [1e-3, 1e-4]
    maxEvals_list = [100, 500, 1000, 5000, 10000]
    min_err = 1
    for lammy in lambda_list:
        for alpha in alpha_list:
            val_err_list = [1]
            for maxEvals in maxEvals_list:
                model = modelConstructor(
                    lammy=lammy, maxEvals=maxEvals, alphaInit=alpha)
                val_err = cross_validate_error(model, X, y, 5)

                if verbose > 0:
                    print(
                        f"Error for lambda={lammy}, maxEvals={maxEvals}, alpha={alpha}: ", val_err)

                if val_err < min_err:
                    min_err = val_err
                    lammy_opt, maxEvals_opt, alpha_opt = lammy, maxEvals, alpha

                tol = 5e-4
                last_val_err = val_err_list[-1:]
                # we expect validation error to decrease as we run more iterations of gradient descent
                # stop if the validation error has not decreased enough to justify long runtime
                if last_val_err - val_err < tol:
                    if verbose > 0:
                        print(
                            "Pruning because validation error did not improve enough...")
                    break
                val_err_list.append(val_err)
    if verbose > 0:
        print(f"Best lambda={lammy}, maxEvals={maxEvals}, alpha={alpha}")

    return lammy_opt, maxEvals_opt, alpha_opt


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
        # we expect validation error to decrease as k grows
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
        lammy, maxEvals, alpha = optimizeLinearClassifierHyper(
            SoftmaxClassifier, X_train_deskewed, y)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model = SoftmaxClassifier(
            lammy=lammy, maxEvals=maxEvals, alphaInit=alpha)
        model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(
            f"Softmax test error for lammy={lammy}, maxEvals={maxEvals}, alpha={alpha}: %.5f: " % test_err)

    elif question == "SVM":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()

        t = time.time()
        lammy, maxEvals, alpha = optimizeLinearClassifierHyper(
            MultiClassSVM, X_train_deskewed, y)
        print("Hyperparameter tuning took %d seconds" % (time.time()-t))

        t = time.time()
        model = MultiClassSVM(lammy=lammy, maxEvals=maxEvals,
                              alphaInit=alpha, verbose=1)
        model.fit(X_train_deskewed, y)
        print("Fitting took %d seconds" % (time.time()-t))

        y_pred = model.predict(X_test_deskewed)
        test_err = np.mean(y_pred != ytest)
        print(
            f"SVM test error for lammy={lammy}, maxEvals={maxEvals}, alpha={alpha}: %.5f: " % test_err)

    else:
        print("Unknown question: %s" % question)
