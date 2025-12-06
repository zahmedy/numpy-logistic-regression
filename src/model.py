import numpy as np


class LogisticRegression:
    def __init__(self, n_features, lr=0.01, reg_lambda=0.001, l2=None):
        """
        Binary logistic regression using gradient descent and optional L2 regularization.

        Parameters
        ----------
        n_features : int
            Number of input features.
        lr : float
            Learning rate for gradient descent.
        reg_lambda : float
            L2 regularization strength. If `l2` is provided, it overrides this value.
        l2 : float | None
            Alias for reg_lambda to keep notebooks backwards-compatible.
        """
        # Preserve backward compatibility with older notebook parameter name
        if l2 is not None:
            reg_lambda = l2

        self.weights = np.random.normal(0, 1, size=(n_features, 1))
        self.bias = 0.0
        self.lr = lr
        self.reg_lambda = reg_lambda

    def forward(self, X):
        z = X @ self.weights + self.bias
        y_hat = 1 / (1 + np.exp(-z))
        return z, y_hat

    def loss(self, y, y_hat):
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        ce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        l2_term = (self.reg_lambda / (2 * len(y))) * np.sum(self.weights ** 2)
        return ce + l2_term

    def backward(self, X, y, y_hat):
        error = y_hat - y
        dW = (X.T @ error) / len(X) + (self.reg_lambda / len(X)) * self.weights
        dB = np.mean(error)
        return dW, dB

    def update(self, dW, dB):
        self.weights -= self.lr * dW
        self.bias -= self.lr * dB

    def accuracy(self, y, y_hat):
        preds = (y_hat >= 0.5).astype(int)
        return (preds == y).mean()


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)
