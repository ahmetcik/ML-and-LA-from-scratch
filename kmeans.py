import numpy as np

class KMeans(object):
    def __init__(self, k=2, max_iter=100, init='kmeans++'):
        self.k = k
        self.max_iter = max_iter
        self.init = init
    
    def _get_label_with_min_distance(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        return distances.argmin(axis=1)

    def _get_init_kmeans_plus(self, X):
        #TODO check this since it should work better
        centroids = np.array([X[np.random.choice(range(X.shape[0]))]])
        for i in range(self.k-1):
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2).sum(1)

            pdf = distances / distances.sum()
            centroid_new = X[np.random.choice(range(X.shape[0]), p=pdf)]
            centroids = np.row_stack((centroids, centroid_new))
        return np.array(centroids)

    def _get_init_random(self, X):
        return np.array([X[np.random.choice(range(n_samples), replace=False)] for k in range(self.k)])

    def _get_init_centroids(self, X):
        if self.init == 'kmeans++':
            return self._get_init_kmeans_plus(X)
        elif self.init == 'random':
            return self._get_init_random(X)

    def fit(self, X):
        # random init of centroids
        n_samples, n_features = X.shape
        centroids = self._get_init_centroids(X)
        
        clusters = [[] for _ in range(self.k)]
        for i in range(self.max_iter):
            labels = self._get_label_with_min_distance(X, centroids)

            for i, x in enumerate(X):
                clusters[labels[i]].append(x)
            
            old_centroids = centroids
            centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            if (centroids == old_centroids).all():
                break
            
        self.centroids = centroids

    def predict(self, X):
        return self._get_label_with_min_distance(X, self.centroids)
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from itertools import product
    random_state = 0
    centers = [[10,7],[0,3],[-5,0],[3,0]]
    n_samples = [500,200,300,100]
    cluster_std=[1.5,1,1,0.5]

    X, y = make_blobs(n_samples=n_samples,
                      random_state=random_state,
                      cluster_std=cluster_std,
                      centers=centers)
    

    km = KMeans(k=4, max_iter=100)
    km.fit(X)
    y = km.predict(X)

    x1min, x1max = X[:, 0].min(), X[:, 0].max()
    x2min, x2max = X[:, 1].min(), X[:, 1].max()

    X_pred = list(product(np.linspace(x1min-0.5, x1max+0.5, 100), 
                          np.linspace(x2min-0.5, x2max+0.5, 100)
                          ))
    X_pred = np.array(X_pred)
    y_pred = km.predict(X_pred)


    plt.scatter(X_pred[:, 0], X_pred[:,1], c=y_pred, alpha=.3, linewidths=0.)
    plt.scatter(X[:, 0], X[:,1], c=y, edgecolors='k')
    plt.show()


