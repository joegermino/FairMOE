import numpy as np

def _hinge_loss(w, X, y):
    yz = y * np.dot(X, w)  # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

    return sum(yz)


def _logistic_loss(w, X, y, return_arr=None):
    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr == True:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out


def _logistic_loss_l2_reg(w, X, y, lam=None):
    if lam is None:
        lam = 1.0

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz))
    l2_reg = (float(lam) / 2.0) * np.sum([elem * elem for elem in w])
    out = logistic_loss + l2_reg
    return out


def log_logistic(X):
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out