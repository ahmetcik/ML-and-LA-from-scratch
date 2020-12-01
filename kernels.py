import numpy as np

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def gaussian_kernel(X1, X2, sigma=1.):
    X_cart_diff = X1[:, np.newaxis, :] - X2
    return np.exp(-0.5 * np.linalg.norm(X_cart_diff, axis=2)**2 / sigma**2)

