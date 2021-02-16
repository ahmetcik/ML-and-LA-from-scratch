import numpy as np
from decision_tree import DecisionTree

class RandomForest(object):
    
    def __init__(self, n_estimators=50, max_features=None, max_depth=2):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features

        self.trees = [DecisionTree(max_depth=max_depth) for _ in range(n_estimators)]

    def fit(self, X, y):
        self.n_classes = np.unique(y).size
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = X.shape[1]
        
        self.indices_features = []
        for i in range(self.n_estimators):
            I_samples  = np.random.choice(range(n_samples), size=n_samples, replace=True)
            I_features = np.random.choice(range(n_features), size=self.max_features)
            self.indices_features.append(I_features)

            X_random = X[np.ix_(I_samples, I_features)]
            y_random = y[I_samples]
            self.trees[i].fit(X_random, y_random)

    def predict(self, X):
        Y = [tree.predict(X[:, self.indices_features[i]]) for i, tree in enumerate(self.trees)]
        # get most common prediction among all estimors
        self.probability = np.array([np.bincount(preds, minlength=self.n_classes) 
                                     for preds in np.transpose(Y)]) / self.n_estimators
        return self.probability.argmax(axis=1)

            
        
    
if __name__ == "__main__":
    from itertools import product
    import matplotlib.pyplot as plt 
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    dt = RandomForest(n_estimators=20, max_depth=2, max_features=2)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    
    print('Training accuracy:', (y == y_pred).sum() / y.size)
    
    X_plot = X[:, [2,3]]
    x1min, x1max = X_plot[:, 0].min(), X_plot[:, 0].max()
    x2min, x2max = X_plot[:, 1].min(), X_plot[:, 1].max()

    X_plot = list(product(np.linspace(x1min-0.5, x1max+0.5, 100), 
                          np.linspace(x2min-0.5, x2max+0.5, 100)
                          ))
    X_plot = np.array(X_plot)
    X_pred = np.transpose([np.ones(len(X_plot)) * X[:, 0].mean(),
                           np.ones(len(X_plot)) * X[:, 1].mean(),
                           X_plot[:, 0],
                           X_plot[:, 1]
                           ])
    y_pred = dt.predict(X_pred)


    plt.scatter(X_plot[:, 0], X_plot[:,1], c=y_pred, alpha=.3, linewidths=0.)
    plt.scatter(X[:, 2], X[:,3], c=y, edgecolors='k')
    plt.show()




