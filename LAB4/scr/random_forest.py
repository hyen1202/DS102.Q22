import numpy as np
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=7, max_depth=7, min_samples=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            dataset_sample = self.bootstrap_samples(dataset)
            X_sample, y_sample = dataset_sample[:, :-1], dataset_sample[:, -1]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self
    
    def bootstrap_samples(self, dataset):
        n_samples = dataset.shape[0]
        np.random.seed(1)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        dataset_sample = dataset[indices]
        return dataset_sample
    
    def most_common_label(self, y):
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(predictions, 0, 1)
        majority_predictions = np.array([self.most_common_label(pred) for pred in preds])
        return majority_predictions