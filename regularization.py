import numpy as np


def none(t):
    return 0

def l1(t):
    return np.sum(np.abs(t))

def J_l1(t):
    return np.sign(t)

def l2(t):
    return 0.5 * np.sum(t ** 2)

def J_l2(t):
    return t

regularizations = {}
regularizations["none"] = none
regularizations["J_none"] = none
regularizations["l1"] = l1
regularizations["J_l1"] = J_l1
regularizations["l2"] = l2
regularizations["J_l2"] = J_l2
