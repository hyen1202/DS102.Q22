import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SVM:
    def __init__(self, C = 10, lr = 0.001, epoches = 100):
        self.C = C
        self.lr = lr
        self.epoches = epoches
        self.w = None
        self.b = None
        self.loss_list = []

    def hinge_loss(self, y, z):
        delta = 1 - y*z # (N,)*(N,)
        return 0.5*np.sum(self.w**2) + self.C * np.maximum(0, delta).sum()

    def predict_score(self, X):
        z = X @ self.w + self.b  # (N, )
        return z
    
    def fit(self, X, y):
        N, dim = X.shape
        self.w = np.zeros(dim)
        self.b = 0.0

        pbar = tqdm(range(self.epoches))
        for i in pbar:
            indices = np.arange(N)
            np.random.seed(42) 
            np.random.shuffle(indices)

            for ith in indices:
                x_i = X[ith] # (dim, )
                y_i = y[ith]

                condition = y_i*(x_i @ self.w + self.b) 

                if condition >= 1:
                    dw = self.w
                    db = 0.0
                else:
                    dw = self.w - self.C * y_i * x_i
                    db = - self.C * y_i
                
                self.w -= self.lr * dw
                self.b -= self.lr * db

            z = self.predict_score(X)
            loss = self.hinge_loss(y, z)
            self.loss_list.append(loss)

            pbar.set_postfix({'loss' : loss})

    def predict(self, X):
        z = X @ self.w + self.b
        return np.sign(z)
    
    def evaluate(self, y, y_hat):
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    