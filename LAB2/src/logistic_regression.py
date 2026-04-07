import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def sigmoid(self, z: np.ndarray):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X: np.ndarray):
        z = X @ self.w
        y_hat = self.sigmoid(z)
        return y_hat
    
    def loss_function(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        epsilon = 1e-9
        l = (1-y)*np.log(1-y_hat + epsilon) + y*np.log(y_hat + epsilon)

        return -l.mean()    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, 1))

        pbar = tqdm(range(self.epoch))

        for e in pbar:
            y_hat = self.predict(X) # (N, 1)
            y = y.reshape(-1, 1)
            delta_y = y_hat - y # (N, 1)

            gradient = (1/N) * delta_y.T @ X # (1, d)
            self.w = self.w - self.lr*gradient.T

            l = self.loss_function(y, y_hat)
            self.losses.append(l)

            pbar.set_description(f"Epoch {e}")
            pbar.set_postfix(loss=l)

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }



