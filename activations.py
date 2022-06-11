import numpy as np


# Assumption: every activation function has the same dimension on the input and output => all jacobians are square matrices
# normal functions take a minibatch-matrix as input, jacobian functions take a vector as input

def identity(z):
    return z.copy()

def J_identity(z):
    return np.eye(z.shape[0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def J_sigmoid(z):
    return np.diag(sigmoid(z) * (1 - sigmoid(z)))

def tanh(z):
    return np.tanh(z)

def J_tanh(z):
    return np.diag(1 / (np.cosh(z) ** 2))

def relu(z):
    return np.maximum(z, 0)

def J_relu(z):
    y = np.zeros(z.shape)
    y[z > 0] = 1
    return np.diag(y)

def softmax(z):
    if z.ndim == 1: # Prevent indexerrors
        z.shape = (z.shape[0], 1)
    y = np.zeros(z.shape)
    for i in range(z.shape[1]):
        y[:, i] = np.exp(z[:, i] - np.max(z[:, i])) # subtract np.max() to prevent overflows, does not change the final value
        y[:, i] /= np.sum(y[:, i])
    return y

def J_softmax(z):
    y = np.zeros((z.shape[0], z.shape[0]))
    s = softmax(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            if i == j:
                y[i, j] = s[i] * (1 - s[i])
            else:
                y[i, j] = - s[i] * s[j]
    return y

activations = {}
activations["identity"] = identity
activations["J_identity"] = J_identity
activations["sigmoid"] = sigmoid
activations["J_sigmoid"] = J_sigmoid
activations["tanh"] = tanh
activations["J_tanh"] = J_tanh
activations["relu"] = relu
activations["J_relu"] = J_relu
activations["softmax"] = softmax
activations["J_softmax"] = J_softmax
