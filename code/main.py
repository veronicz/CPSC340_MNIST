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
from mlp_optimizer import optimize
from neural_net import NeuralNet


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "KNN":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()

        # 5-fold cross-validation
        min_err = 1
        k_opt = 1
        k_list = range(1, 30)
        valid_err_list = [1]
        for k in k_list:
            valid_err = cross_validate_error(KNN(k=k), X_train_deskewed, y, 5)
            if (valid_err <= min_err):  # choose the simper model if error is the same
                min_err = valid_err
                k_opt = k

            valid_err_list.append(valid_err)
            last_valid_errs = valid_err_list[-5:]
            # we expect validation error to decrease as k grows
            # stop if the validation error has been increasing for the last 5 k's
            if (last_valid_errs == sorted(last_valid_errs)):
                break  # model is becoming too simple

        model = KNN(k=k_opt)
        model.fit(X_train_deskewed, y)
        y_pred = model.predict(X_test_deskewed)
        test_error = np.mean(y_pred != ytest)
        print("KNN test error for k=%d: %.5f" % (k_opt, test_error))

    elif question == "MLP":
        X_train_deskewed, y, X_test_deskewed, ytest = loadDeskewedMNIST()
        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        # hu, batch_size, epochs, lammy, lr = optimize(X_train_deskewed, Y, 5)
        valid_err_list = [1]
        hu_list = [500, 700]
        bs_list = [256, 512, 1024]
        epochs_list = [100, 300, 500]
        lambda_list = [0.1, 1e-3]
        lr_list = [10**-4, 10**-3]
        min_err = 1
        hu_opt, bs_opt, epochs_opt, lammy_opt, lr_opt = 0, 0, 0, 0, 0
        for hu in hu_list:
            for bs in bs_list:
                for epochs in epochs_list:
                    for lammy in lambda_list:
                        for lr in lr_list:
                            print("begin:", hu, bs, epochs, lammy, lr)
                            model = NeuralNet([hu], lammy, lr, bs, epochs)
                            # val_err = cross_validate_error(
                            #     model, X_test_deskewed, Y, 1)
                            model.fit_SGD(
                                X_train_deskewed[:40000, :], Y[:40000, :])
                            y_pred = model.predict(X_train_deskewed[40000:, :])
                            val_err = np.mean(y_pred != y[40000:])
                            print("end:", val_err)
                            if (val_err < min_err):  # choose the simper model if error is the same
                                min_err = val_err
                                hu_opt, bs_opt, epochs_opt, lammy_opt, lr_opt = hu, bs, epochs, lammy, lr
                                print(val_err, hu_opt, bs_opt,
                                      epochs_opt, lammy_opt, lr_opt)

        # hidden_layer_sizes = [600]
        # model = NeuralNet(hidden_layer_sizes, lammy=lammy,
        #                   learning_rate=lr, batch_size=256)

        # t = time.time()
        # model.fit_SGD(X_train_deskewed, Y)
        # print("Fitting took %d seconds" % (time.time()-t))

        # # Compute test error
        # yhat = model.predict(X_test_deskewed)
        # testError = np.mean(yhat != ytest)
        # print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)
