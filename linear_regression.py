import numpy as np
from linear_algebra import svd
class LinearRegressor(object):
    """ Class for computing least squares solution if lambda_l2 is zero,
        otherwise ridge regression."""

    def __init__(self, lambda_l2=0., intercept=True):
        self.intercept = intercept
        self.lambda_l2 = lambda_l2

    def fit(self, X, Y):
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
        
        U, S, V = svd(X)
        
        # set diagonal D depending on if least squares or ridge regression
        if self.lambda_l2 == 0:
            D = 1 / S    
        else:
            D = S / (S**2 + self.lambda_l2)
        
        self.coefs = V * D @ U.T @ Y
            
    def predict(self, X):
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
        return np.dot(X, self.coefs)

    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

if __name__ == '__main__':
    from sklearn import datasets
    from plot import scatter
    from normalize import get_standardized

    # Load the diabetes dataset
    X, Y = datasets.load_diabetes(return_X_y=True)
    
    # Split the data into training/testing sets
    X_train, X_test = X[:-20], X[-20:]
    Y_train, Y_test = Y[:-20], Y[-20:]
    X_train, X_test = get_standardized(X_train, X_test)
    
    regr = LinearRegressor(lambda_l2=0.001)
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    
    scatter(Y_test, Y_pred)
    
