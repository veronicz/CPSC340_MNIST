import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

import utils
from backward import *
from forward import forward


class CNN(BaseEstimator, ClassifierMixin):
    '''
    The network uses two consecutive convolutional layers followed by a max pooling operation to extract features from the input image. After the max pooling operation, the representation is flattened and passed through a Multi-Layer Perceptron (MLP) to carry out the task of classification.
    '''

    def __init__(self, n1_filters=8, n2_filters=8, filter_size=5, conv_stride=1, mlp_size=128, beta1=0.95, beta2=0.99, learning_rate=0.01, epochs=2, batch_size=32, img_dim=28, img_depth=1):
        self.n1_filters = n1_filters
        self.n2_filters = n2_filters
        self.filter_size = filter_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_dim = img_dim
        self.img_depth = img_depth
        self.beta1 = beta1
        self.beta2 = beta2
        self.conv_stride = conv_stride
        self.mlp_size = mlp_size

    def fit(self, X, Y):
        train_data = np.hstack((X, Y))

        nf1, nf2 = self.n1_filters, self.n2_filters
        f, n, s = self.filter_size, self.img_dim, self.conv_stride
        mlp_size = self.mlp_size
        pool_dim = (((n - f)//s + 1 - f)//s + 1 - 2)//2 + 1

        f1, f2 = (nf1, self.img_depth, f, f), (nf2, nf1, f, f)
        falttened_pool_size = pool_dim * pool_dim * nf2
        # output 10 classes
        w3, w4 = (mlp_size, falttened_pool_size), (10, mlp_size)

        f1 = utils.initializeFilter(f1)
        f2 = utils.initializeFilter(f2)
        w3 = utils.initializeWeight(w3)
        w4 = utils.initializeWeight(w4)

        b1 = np.zeros((f1.shape[0], 1))
        b2 = np.zeros((f2.shape[0], 1))
        b3 = np.zeros((w3.shape[0], 1))
        b4 = np.zeros((w4.shape[0], 1))

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
        cost = []
        for epoch in range(self.epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + self.batch_size]
                       for k in range(0, train_data.shape[0], self.batch_size)]

            t = tqdm(batches)
            for x, batch in enumerate(t):
                params, cost = adamGD(batch, params, cost, self.img_depth,
                                      self.img_dim, self.conv_stride, self.beta1, self.beta2, self.learning_rate)
                t.set_description("Cost: %.2f" % (cost[-1]))

        self.params = params
        return params

    def predict(self, X, pool_f=2, pool_s=2):
        '''
        Make predictions with trained filters/weights.
        '''
        n = X.shape[0]
        t = tqdm(range(n), leave=True)
        X = X.reshape(n, self.img_depth, self.img_dim, self.img_dim)
        y_pred = np.zeros(n)
        for i in t:
            x = X[i]
            probs = forward(x, self.params, self.conv_stride,
                            pool_f, pool_s)[0]
            y_pred[i] = np.argmax(probs)
        return y_pred

    def predict_proba(self, X, pool_f=2, pool_s=2):
        n = X.shape[0]
        X = X.reshape(n, self.img_depth, self.img_dim, self.img_dim)
        probs = np.zeros((n, 10))
        for i in range(n):
            x = X[i]
            probs[i] = forward(x, self.params, self.conv_stride, pool_f, pool_s)[
                0].ravel()
        return probs


def adamGD(batch, params, cost, img_depth, img_dim, conv_s, beta1, beta2, learning_rate):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = batch[:, 0:-1]  # get batch inputs
    X = X.reshape(len(batch), img_depth, img_dim, img_dim)
    Y = batch[:, -1]  # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum,RMS params
    df1, v1, s1 = (np.zeros(f1.shape) for i in range(3))
    df2, v2, s2 = (np.zeros(f2.shape) for i in range(3))
    dw3, v3, s3 = (np.zeros(w3.shape) for i in range(3))
    dw4, v4, s4 = (np.zeros(w4.shape) for i in range(3))
    db1, bv1, bs1 = (np.zeros(b1.shape) for i in range(3))
    db2, bv2, bs2 = (np.zeros(b2.shape) for i in range(3))
    db3, bv3, bs3 = (np.zeros(b3.shape) for i in range(3))
    db4, bv4, bs4 = (np.zeros(b4.shape) for i in range(3))

    for i in range(batch_size):
        x = X[i]
        y = np.eye(10)[int(Y[i])].reshape(10, 1)

        grads, loss = conv(x, y, params, conv_s, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1 += df1_
        db1 += db1_
        df2 += df2_
        db2 += db2_
        dw3 += dw3_
        db3 += db3_
        dw4 += dw4_
        db4 += db4_

        cost_ += loss

    # Parameter Update
    def updateAdamParams(v, s, d, f):
        v = beta1*v + (1-beta1)*d/batch_size  # momentum update
        s = beta2*s + (1-beta2)*(d/batch_size)**2  # RMSProp update
        # combine momentum and RMSProp to perform update with Adam
        f -= learning_rate * v/np.sqrt(s+1e-7)
        return v, s, f

    v1, s1, f1 = updateAdamParams(v1, s1, df1, f1)
    bv1, bs1, b1 = updateAdamParams(bv1, bs1, db1, b1)

    v2, s2, f2 = updateAdamParams(v2, s2, df2, f2)
    bv2, bs2, b2 = updateAdamParams(bv2, bs2, db2, b2)

    v3, s3, w3 = updateAdamParams(v3, s3, dw3, w3)
    bv3, bs3, b3 = updateAdamParams(bv3, bs3, db3, b3)

    v4, s4, w4 = updateAdamParams(v4, s4, dw4, w4)
    bv4, bs4, b4 = updateAdamParams(bv4, bs4, db4, b4)

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    return params, cost


def conv(image, label, params, conv_s, pool_f, pool_s):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    ################################################
    ############## Forward Operation ###############
    ################################################
    probs, z, fc, pooled, conv2, conv1 = forward(
        image, params, conv_s, pool_f, pool_s)

    ################################################
    #################### Loss ######################
    ################################################

    # categorical cross-entropy loss
    loss = utils.categoricalCrossEntropy(probs, label)

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
    # loss gradient of final dense layer biases
    db4 = np.sum(dout, axis=1).reshape(b4.shape)

    dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
    dz[z <= 0] = 0  # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis=1).reshape(b3.shape)

    # loss gradients of fully-connected layer (pooling layer)
    dfc = w3.T.dot(dz)
    # reshape fully connected into dimensions of pooling layer
    dpool = dfc.reshape(pooled.shape)

    # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2 = maxpoolBackward(dpool, conv2, pool_f, pool_s)
    dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

    # backpropagate previous gradient through second convolutional layer.
    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_s)
    dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

    # backpropagate previous gradient through first convolutional layer.
    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_s)

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss
