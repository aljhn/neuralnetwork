import numpy as np


def mse(y_pred, y_true):
    return np.sum((y_pred - y_true)**2, axis=0) / y_pred.shape[0]

# Dont sum the vector because the derivative with respect to an element of the vector is the only one left
def J_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[0]

def cross_entropy(y_pred, y_true): # finn ut av basen, bruk np.log2() istedet hvis, UPDATE: log2 gjør det vanskeligere å derivere så ikke
    return - np.sum(np.log(y_pred + 1e-10) * y_true, axis=0) # "*" does elementwise multiplication

def J_cross_entropy(y_pred, y_true):
    return - 1 / (y_pred + 1e-10) * y_true

losses = {}
losses["mse"] = mse
losses["J_mse"] = J_mse
losses["cross_entropy"] = cross_entropy
losses["J_cross_entropy"] = J_cross_entropy
