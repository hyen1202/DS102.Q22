import numpy as np

class Kmeans:
    def __init__(self, K):
        self.K = K

    def initialize_centroids(self, X):
        randomized_X = np.random.permutation(X.shape[0])
        centroid_idx = randomized_X[: self.K]
        self.centroids = X[centroid_idx]

    def assign_points_centroids(self, X):
        X = np.expand_dims(X, axis=1)
        distance = np.linalg.norm((X - self.centroids), axis=-1)
        points = np.argmin(distance, axis=1)
        return points
    
    def compute_mean(self, X, points):
        centroids = np.zeros((self.K, X.shape[1]))
        for i in range(self.K):
            centroid_mean = X[points == i].mean(axis=0)
            centroids[i] = centroid_mean
        return centroids
    
    def fit(self, X, iterations=10):
        self.initialize_centroids(X)
        for i in range(iterations):
            points = self.assign_points_centroids(X)
            self.centroids = self.compute_mean(X, points)
        return self.centroids, points