import numpy as np

class DecisionTree(object):
    
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.current_depth = 0
        self.tree = self._get_tree(X, y)

    def _get_tree(self, X, y):
        n_samples, n_features = X.shape
        
        cost_best = np.inf
        for i_feature in range(n_features):
            x = X[:, i_feature] 
            for decision_boundary in x:
                mask = x <= decision_boundary 

                X1, X2 = X[mask], X[~mask]
                if len(X1) > 0 and len(X2) > 0:
                    upper_left, lower_right = x[mask].max(), x[~mask].min()
                    decision_boundary = 0.5 * (upper_left + lower_right)
                    y1, y2 = y[mask], y[~mask]
                    p = mask.sum() / n_samples

                    cost = p * self.get_gini(y1) + (1. - p) * self.get_gini(y2)
                    if cost < cost_best:
                        cost_best = cost
                        i_feature_best = i_feature
                        decision_boundary_best = decision_boundary
                        arrays_best = [X1, X2, y1, y2]

                        

        if cost_best < np.inf and self.current_depth < self.max_depth and  np.unique(y).size > 1:
            X1, X2, y1, y2 = arrays_best
            true_branch = self._get_tree(X1, y1)
            false_branch = self._get_tree(X2, y2)
            return i_feature_best, decision_boundary_best, true_branch, false_branch, None
        
        most_common_label = np.bincount(y).argmax()
        return None, None, None, None, most_common_label

    def predict(self, X):
        y = [self.classify(x) for x in X]
        return np.array(y)
    
    def classify(self, x, tree=None):
        if tree is None:
            tree = self.tree
        i_feature, decision_boundary, true_branch, false_branch, label = tree

        if label is not None:
            return label

        if x[i_feature] <= decision_boundary:
            return self.classify(x, tree=true_branch)
        else:
            return self.classify(x, tree=false_branch)
        

    def get_gini(self, y):
        gini = 1.
        for label in np.unique(y):
            p = (y == label).sum() / y.size
            gini -= p**2
        return gini

        
    
if __name__ == "__main__":
    from itertools import product
    import matplotlib.pyplot as plt 
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    dt = DecisionTree()
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




