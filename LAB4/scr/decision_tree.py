import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
        left = []
        right = []

        for row in dataset:
            if row[feature] <= threshold:
                left.append(row)
            else:
                right.append(row)

        left = np.array(left)
        right = np.array(right)
        return left, right

    def entropy(self, y):
        entropy = 0
        labels = np.unique(y)
        for label in labels: 
            label_examples = y[y == label]
            p = len(label_examples) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def information_gain(self, parent, left, right):
        IG = 0
        parrent_e = self.entropy(parent)

        w_left = len(left) / len(parent)
        w_right = len(right) / len(parent)

        entropy_after_split = w_left*self.entropy(left) + w_right*self.entropy(right)

        IG = parrent_e - entropy_after_split
        return IG

    def best_split(self, dataset, num_samples, num_features):
        best_split = {
            'gain': -1,
            'feature': None,
            'threshold': None
        }

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left, right = self.split_data(dataset, feature_index, threshold)
                if len(left) and len(right):
                    y, y_left, y_right = dataset[:, -1], left[:, -1], right[:, -1]
                    IG = self.information_gain(y, y_left, y_right)

                    if IG > best_split['gain']:
                        best_split['feature'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left'] = left
                        best_split['right'] = right
                        best_split['gain'] = IG
        return best_split

    def calculate_leaf_value(self, y):
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value

    def build_tree(self, dataset, current_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_split = self.best_split(dataset, n_samples, n_features)
        
            if best_split['gain'] > 0:
                left_node = self.build_tree(best_split['left'], current_depth+1)
                right_node = self.build_tree(best_split['right'], current_depth+1)
                return Node(best_split['feature'], best_split['threshold'], 
                            left_node, right_node, best_split["gain"])
                
        leaf_value = self.calculate_leaf_value(y)

        return Node(value=leaf_value)

    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

    def make_prediction(self, x, node):
        if node.value is not None:
            return node.value
        else:
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)