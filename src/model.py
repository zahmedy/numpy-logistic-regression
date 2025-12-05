import numpy as np

class LogisticRegression:
    def __init__(self, n_features , lr=0.01, l2=0.0):
        # random weights
        self.weights = np.random.normal(0, 1, size=(n_features, 1))
        # bias
        self.bias = 0
        # learning rate
        self.lr = lr
        # L2 regularization 
        self.l2 = l2

    def forward(self, X):
        z = X @ self.weights + self.bias
        sigmoid = 1 / (1 + np.exp(-z))
        return z, sigmoid

    def loss(self, y, y_hat):
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss.mean()

    def backward(self, X, y, y_hat):
        error = y_hat - y          # (N, 1)
        dW = X.T @ error           # (D, 1)
        dB = np.sum(error)         # scalar
        return dW, dB

    def update(self, dW, dB):
        dW += self.l2 * self.weights
        self.weights -= self.lr * dW 
        self.bias -= self.lr * dB
    
    def accuracy(self, y, y_hat):
        preds = (y_hat >= 0.5).astype(int)
        return (preds == y).mean()

    class StandardScaler:
        def fit(self, X):
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        def transform(self, X):
            return (X - self.mean) / (self.std + 1e-8)
    
