import numpy as np
from quadratic_programming import qp
class NonNegativeLinearRegressor(object):
    """ Class for computing nonnegative least squares solution if lambda_l2 is zero,
        otherwise nonnegative ridge regression."""

    def __init__(self, lambda_l2=0., intercept=True, **kwargs):
        self.intercept = intercept
        self.lambda_l2 = lambda_l2
        self.qp_kwargs = kwargs

    def fit(self, X, Y, **kwargs):
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
        
        Q = X.T @ X
        p = - X.T @ Y

        if self.lambda_l2 > 0:
            Q += self.lambda_l2 * np.eye(X.shape[1])
        
        # Apart from the constraint coefs => 0, which is already implemented as 
        # default in the qp function, no further constraint is needed
        G = np.empty((0, X.shape[1]))
        h = np.empty((0, ))
        A = np.empty((0, X.shape[1]))
        b = np.empty((0, ))

        self.coefs = qp(Q, p, G, h, A, b, **self.qp_kwargs)
            
    def predict(self, X):
        if self.intercept:
            X = np.insert(X, -1, 1., axis=1)
        return np.dot(X, self.coefs)

if __name__ == '__main__':
    from sklearn import datasets
    from plot import scatter
    
    # Load the diabetes dataset
    X, Y = datasets.load_diabetes(return_X_y=True)
    
    # Split the data into training/testing sets
    X_train, X_test = X[:-20], X[-20:]
    Y_train, Y_test = Y[:-20], Y[-20:]
    
    regr = NonNegativeLinearRegressor(lambda_l2=0.001,
                           max_steps_qp=5000, alpha=0.001, tol_eigen_value=1e-1, mu=0.000001 # qp parameters
                           )
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    print("coefficients:")
    print(regr.coefs)
    scatter(Y_test, Y_pred)
    
