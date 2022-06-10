import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y * (1.0 - y)


def dtanh(x):
    return 1.0 - np.tanh(x) ** 2


def lrelu(x, alpha=1e-2):
    return np.maximum(x, x * alpha)


def dlrelu(x, alpha=1e-2):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
