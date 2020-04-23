import numpy as np


#####################################################
################ Forward Operations #################
#####################################################
def forward(image, params, conv_s, pool_f, pool_s):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    conv1 = convolution(image, f1, b1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    # second convolution operation
    conv2 = convolution(conv1, f2, b2, conv_s)
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer

    # predict class probabilities with the softmax activation function
    probs = softmax(out)
    return probs, z, fc, pooled, conv2, conv1


def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
    if len(image.shape) < 3:
            print(image.shape)
    n_c, in_dim, _ = image.shape  # image dimensions

    out_dim = (in_dim - f)//s+1  # calculate output dimensions

    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(
                    filt[curr_f] * image[:, curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out


def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, h_prev, w_prev = image.shape

    h = (h_prev - f)//s+1
    w = (w_prev - f)//s+1

    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(
                    image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled


def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)
