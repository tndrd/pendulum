import numpy as np
import matplotlib.pyplot as plt

DERIV_H = 1e-2
FIT_TOLERANCE = 1e-3
MAXIMUM_STEPS = 10**4

def _partial_derivative(func, i, X, h=DERIV_H):
    dX = np.zeros_like(X)
    dX[i] = h
    return (func(X + dX) - func(X - dX)) / (2 * DERIV_H)

def _jacob(func, X, h=DERIV_H):
    N = len(X)
    J = np.zeros([N, N])

    for i in range(N):
        J[:, i] = _partial_derivative(func, i, X, h)
    
    return J

def newton(func, band, X0=None, h=DERIV_H):
    if X0 is None:
        x0 = np.random.uniform(*band[0])
        y0 = np.random.uniform(*band[1])
        X0 = np.array([x0, y0])

    X = X0

    for _ in range(MAXIMUM_STEPS):
        res = func(X)
        if np.linalg.norm(res) < FIT_TOLERANCE:
            return X
        
        J = _jacob(func, X)
        #dX = np.linalg.lstsq(J, -res, rcond=None)[0]
        dX = np.linalg.solve(J, -res)
        X += dX

    raise RuntimeError("Reached max step count")