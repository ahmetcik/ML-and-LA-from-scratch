import numpy as np

class LogisticRegression(object):
    
    def __init__(self, intercept=True, alpha=0.01, max_steps=2000,):
        self.intercept = intercept
        self.alpha = alpha # learning rate
        self.max_steps = max_steps

    def fit(self, X, Y): 
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
    
        coefs = np.zeros(X.shape[1])
        for _ in range(self.max_steps):
            pred = self.sigmoid(X @ coefs)
            gradient = np.dot(X.T, (pred - Y)) 
            coefs -= self.alpha * gradient
        
        self.coefs = coefs
    def predict(self, X): 
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
        return self.sigmoid(X @ self.coefs).round().astype(int)

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

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
    Y = np.array([0, 0, 0, 0, 1,1,1,1])

    log = LogisticRegression()
    log.fit(X, Y)
    y_pred = log.predict(X)
    

    X_pred = list(product(np.linspace(0, 2, 100), np.linspace(0, 3, 100)))
    X_pred = np.array(X_pred)
    Y_pred = log.predict(X_pred)

    plt.scatter(X_pred[:, 0], X_pred[:,1], c=Y_pred, alpha=.3, linewidths=0.)
    plt.scatter(X[:, 0], X[:,1], c=Y, edgecolors='k')
    plt.show()

