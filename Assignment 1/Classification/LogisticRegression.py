import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)


from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogisticRegression:

    
    def __init__(self, max_epochs=1000, learning_rate=0.05):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y,learning_rate):
        self.learning_rate = learning_rate
        num_samples, num_features = X.shape

        y_reshape = y.reshape(-1, 1)
        y = one_hot_encoder.fit_transform(y_reshape)

        outputs = y.shape[1]

        self.weights = np.zeros((num_features, outputs))
        self.bias = 0
        
        for _ in range(self.max_epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(y_pred)

            dw = 1/num_samples * np.dot(X.T, (y_pred - y))
            db = 1/num_samples * np.sum(y_pred - y, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(y_pred)
        return np.argmax(y_pred, axis=1)
    