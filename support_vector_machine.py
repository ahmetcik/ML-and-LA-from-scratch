import numpy as np
from quadratic_programming import qp
from kernels import linear_kernel, gaussian_kernel

class SupportVectorMachine(object):
    def __init__(self, C=1.0, kernel=linear_kernel, sigma=None, tol_alpha=1e-4, **kwargs):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.tol_alpha = tol_alpha
        self.qp_kwargs = kwargs
    
    def get_kernel(self, X1, X2):
        try:
            return self.kernel(X1, X2, sigma=self.sigma)
        except:
            return self.kernel(X1, X2)

    def fit(self, X, Y):
        n = X.shape[0]

        K = self.get_kernel(X, X)
        Q = np.outer(Y, Y) * K
        p = - np.ones(n)


        # alpha => is already implemented by default in qp.
        # Therfore only alpha <= C is needed.
        G = np.eye(n)
        h = np.ones(n) * self.C
        
        # sum_i y_i alpha_i = 0
        A = Y[np.newaxis]
        b = 0.
        
        # solve dual problem via quadratic programming
        alphas = qp(Q, p, G, h, A, b, **self.qp_kwargs)
        
        self.idx_support = np.where(alphas > self.tol_alpha)[0]
        self.alphas = alphas[self.idx_support]
        self.support_vectors = X[self.idx_support]
        self.support_y = Y[self.idx_support]
        
        self.intercept = np.dot(K[np.ix_(self.idx_support[:1], self.idx_support)][0], 
                                self.alphas* self.support_y)
        self.intercept -= Y[self.idx_support[0]]

    def predict(self, X):
        K = self.get_kernel(X, self.support_vectors)
        pred = np.dot(K, self.alphas* self.support_y) - self.intercept
        return np.sign(pred)

if __name__ == "__main__":
    from itertools import product
    import matplotlib.pyplot as plt
    X = [[0,1.2],
         [1,1],
         [1,0],
         [2,0],
         [0,2],
         [1,2],
         [2,3],
         [0,1],
         ]
    X = np.array(X)
    Y = np.array([-1, -1, -1, -1, 1,1,1,1])

    svm = SupportVectorMachine(kernel=gaussian_kernel, C=10., sigma=1.5, 
                               tol_eigen_value=1e-1, mu=.000001, alpha=0.1, max_steps_qp=200 # 
                               )
    svm.fit(X, Y)
    y_pred = svm.predict(X)
    

    X_pred = list(product(np.linspace(0, 2, 100), np.linspace(0, 3, 100)))
    X_pred = np.array(X_pred)
    Y_pred = svm.predict(X_pred)

    X_support = X[svm.idx_support]

    plt.scatter(X_pred[:, 0], X_pred[:,1], c=Y_pred, alpha=.3, linewidths=0.)
    plt.scatter(X[:, 0], X[:,1], c=Y, edgecolors='k')
    plt.scatter(X_support[:, 0], X_support[:,1], color='k', facecolors='none', edgecolors='k', s=140)
    plt.show()
