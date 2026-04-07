import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.W = None
        self.losses = []

    def loss_function(self, y: np.ndarray, y_hat: np.ndarray):
        return -(y * np.log(y_hat + 1e-15)).sum(axis=-1).mean()

    def softmax(self, z: np.ndarray):
        z = z - np.max(z, axis=1, keepdims=True)
        e_z = np.exp(z)
        return e_z / e_z.sum(axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray):
        z = X @ self.W
        y_hat = self.softmax(z)
        return y_hat
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        _, c = y.shape
        self.W = np.zeros((d, c))

        pbar = tqdm(range(self.epoch))

        for e in pbar:
            y_hat = self.predict(X) # (N, c)
            delta_y = y_hat - y # (N, c)

            gradient = (1/N) * delta_y.T @ X # (c, d)
            self.W = self.W - self.lr*gradient.T

            l = self.loss_function(y, y_hat)
            self.losses.append(l)

            pbar.set_description(f"Epoch {e}")
            pbar.set_postfix(loss=l)

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        if y_hat.ndim == 2:
            y_hat = np.argmax(y_hat, axis=1)

        precision = precision_score(y, y_hat, average='macro', zero_division=0)
        recall = recall_score(y, y_hat, average='macro', zero_division=0)
        f1 = f1_score(y, y_hat, average='macro', zero_division=0)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
