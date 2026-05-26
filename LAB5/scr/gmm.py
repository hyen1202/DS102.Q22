import numpy as np
from tqdm import tqdm

class GMM:
    def __init__(self, n_components, max_iter=100, comp_names=None):
        self.n_components = n_components
        self.max_iter = max_iter

        if comp_names == None:
            self.comp_names = [f'comp{index}' for index in range(self.n_components)]
        else:
            self.comp_names = comp_names

        self.pi = [1/self.n_components for _ in range(self.n_components)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        D = X.shape[0]
        reg_covariance = covariance_matrix + np.eye(D)*1e-6


        normalize_coeff = (2*np.pi)**(-D/2) * np.linalg.det(reg_covariance)**(-1/2)
        quad_form = -np.dot(np.dot((X - mean_vector).T, np.linalg.inv(reg_covariance)), (X - mean_vector)) / 2
        return normalize_coeff * np.exp(quad_form)
    
    def fit(self, X):
        new_X = np.array_split(X, self.n_components)

        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]

        for _ in tqdm(range(self.max_iter)):
            # E step
            self.r = np.zeros((len(X), self.n_components))

            for n in range(len(X)):
                for k in range(self.n_components):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])

                row_sum = sum(self.r[n])
                if row_sum == 0:
                    row_sum = 1e-10  
                
                for k in range(self.n_components):
                    self.r[n][k] /= row_sum  
                    
            # M step
            N = np.sum(self.r, axis=0)
            self.mean_vector = np.zeros((self.n_components, len(X[0])))

            # Update mean vector
            for k in range(self.n_components):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_components)]

            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_components)]

            for k in range(self.n_components):
                self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0)
            #self.covariance_matrixes = [1/N[k] * self.covariance_matrixes[k] for k in range(self.n_components)]

            self.pi = [N[k]/len(X) for k in range(self.n_components)]

    def predict(self, X):
        probas = []
        for n in range(len(X)):
            probas.append([self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k]) for k in range(self.n_components)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster
    
