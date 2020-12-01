from linear_regression import LinearRegressor
import numpy as np

class OrthogonalMatchingPursuit2(object):
    def __init__(self, n_nonzero_coefs=3, intercept=True):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.intercept = intercept

    def fit(self, X, Y): 

        linear_regressor = LinearRegressor(intercept=self.intercept)
        indices_best = []
        R = Y
        for i in range(self.n_nonzero_coefs):
            i_best = abs(np.dot(X.T, R)).argmax()
            indices_best.append(i_best)
            linear_regressor.fit(X[:, indices_best], Y)
            R = Y - linear_regressor.predict(X[:, indices_best]) 

        self.indices_best = indices_best
        self.linear_regressor = linear_regressor

    def predict(self, X):
        return self.linear_regressor.predict(X[:, self.indices_best])

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from normalize import get_standardized
    from plot import scatter

    X, Y = make_regression(noise=4, random_state=0)
    X = get_standardized(X)
    
    omp = OrthogonalMatchingPursuit2(n_nonzero_coefs=10)
    omp.fit(X, Y)
    Y_pred = omp.predict(X)
    

    print("indices of best columns out of 100:", omp.indices_best)
    scatter(Y, Y_pred)

    

    


    
