import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return ...

def problem_1c (A, B, C):
    return ...

def problem_1d (x, y):
    return ...

def problem_1e (A, x):
    return ...

def problem_1f (A, x):
    return ...

def problem_1g (A, i):
    return ...

def problem_1h (A, c, d):
    return ...

def problem_1i (A, k):
    return ...

def problem_1j (x, k, m, s):
    return ...

def problem_1k (A):
    return ...

def problem_1l (x):
    return ...

def problem_1m (x, k):
    return ...

def problem_1n (X):
    return ...

def linear_regression (X_tr, y_tr):
    ...

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
