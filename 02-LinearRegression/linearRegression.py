"""
Credit to Patrick Loeber
"""


import numpy as np

class LinearRegression:
    """ Linear Regression Using Gradient Descent
    Initilization: 
        lr = learning rate 
        niters = number of iterations
        weights = none
        bais =  none
    """

    def __init__(self, learning_rate = 0.01, niters=1000):
        self.lr = learning_rate
        self.niters = niters
        self.weights = None 
        self.bias = None

    def fit(self, X, y):
        """ Fitting the traininng data
        x: n_features
        y: n_target_value
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0

        for _ in range(self.niters):
            y_pred = np.dot(X,self.weights) + self.bias
            residuals = y_pred - y 
            # Compute gradient 
            dw = (1/n_samples) * np.dot(X.T, residuals)
            db = (1/n_samples) * np.sum(residuals)

            #update paraemeters 
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict(self, X):
        y_approx = np.dot(X, self.weights) + self.bias
        return y_approx
