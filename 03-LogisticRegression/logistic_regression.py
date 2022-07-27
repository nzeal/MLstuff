import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate=0.001, niters=1000):
        self.lr = learning_rate
        self.niters = niters
        self.weights = None
        self.bais = None

    def fit(self, X, y):
        nsamples, nfeatures = X.shape

        # init parameter
        self.weights = np.zeros(nfeatures)
        self.bais = 0

        # gradient descent
        for _ in range(self.niters):
            y_predicted = np.dot(X, self.weights) + self.bais
            y_predicted = self._sigmoid(y_predicted)

            residuals = y_predicted - y
            # compute gradients
            dw = (1/nsamples) * np.dot(X.T, residuals)
            db = (1/nsamples) * np.sum(X.T, residuals)

            # update parameters
            self.weights -= self.lr * dw
            self.bais -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bais
        y_predicted = sigmoid(y_predicted)
        # clip to 1or 0
        y_predicted = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted)
