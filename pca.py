import numpy as np
from linear_algebra import svd


class PCA(object):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X):
        self.mean = X.mean(0)
        U, S, V = svd(X - self.mean)
        
        self.components = V.T[:self.n_components]
        self.singular_values = S[:self.n_components]

        explained_variance = (S ** 2) / (len(X) - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()

        self.explained_variance = explained_variance[:self.n_components]
        self.explained_variance_ratio = explained_variance_ratio[:self.n_components]

    def transform(self, X):
        return np.dot(X - self.mean, self.components.T)



if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    pca =  PCA(n_components=2)
    pca.fit(X)
    X_low = pca.transform(X)
    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        plt.scatter(X_low[Y==label, 0], X_low[Y==label, 1], label=name)
    plt.legend()
    plt.show()



