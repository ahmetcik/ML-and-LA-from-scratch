import numpy as np
from linear_algebra import cholesky_decomposition, solve_triangular
from kernels import linear_kernel, gaussian_kernel

class KernelRidgeRegression(object):
    """Kernel ridge regression"""
    def __init__(self, kernel=gaussian_kernel, lambda_l2=0.001, sigma=1):
        self.lambda_l2 = lambda_l2
        self.sigma = sigma
        self.kernel = kernel
    
    def fit(self, X, Y):
        self.X_train = X

        K = self.get_kernel(X, X)
        K_lam = K + self.lambda_l2 * np.eye(K.shape[0])
        
        # Cholesky decomposition
        L = np.linalg.cholesky(K_lam)
        beta        = solve_triangular(  L,    Y, lower=True)
        self.alphas = solve_triangular(L.T, beta, lower=False)
    
    def predict(self, X):
        K = self.get_kernel(X, self.X_train)
        return np.dot(K, self.alphas)
    
    def get_kernel(self, X1, X2):
        try:
            return self.kernel(X1, X2, sigma=self.sigma)
        except:
            return self.kernel(X1, X2) 

        
if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(0)

    X = 5 * rng.rand(10000, 1)
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
    X_test = np.linspace(0, 5, 100000)[:, None]
    train_size = 100

    X_train = X[:train_size]
    Y_train  = y[:train_size]
     
    stan = StandardScaler()
    X_train = stan.fit_transform(X_train)
    X_test  = stan.transform(X_test)
    
    # Of course hyperparamaters lambda_l2 and sigma actually need
    # to be optimized on the training set e.g. on a square grid.
    krr = KernelRidgeRegression(lambda_l2=0.1, sigma=1.)
    krr.fit(X_train, Y_train)
    Y_pred = krr.predict(X_test)
    
    plt.scatter(X_train.flatten(), y[:100], c='k')
    plt.plot(X_test.flatten(), Y_pred, c='g')
    plt.show()

