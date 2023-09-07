import numpy as np
import pickle

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate = 0.001):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.mean_sqrd_error = None

        num_samples, num_features = X.shape
        output_dim = y.shape[1]

        self.weights = np.zeros((num_features, output_dim))
        self.bias = np.zeros((1, output_dim))

        weights = self.weights
        bias = self.bias
        l2 = self.regularization

        value_size = int(0.1 * num_samples)
        X_val = X[:value_size]
        y_val = y[:value_size]
        X_train = X[value_size:]
        y_train = y[value_size:]

        best_loss = np.inf
        count_petience_increase = 0

    
        mse = []

        for epoch in range(self.max_epochs):
            

            for i in range(0, int(0.9 * num_samples), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                y_pred = np.dot(X_batch, weights) + bias
                loss = np.mean(np.square(y_pred - y_batch)) + l2 * np.mean(np.square(weights))

                grad_w = (2/self.batch_size) * np.dot(X_batch.T, y_pred - y_batch) + 2 * l2 * weights
                grad_b = (2/self.batch_size) * np.sum(y_pred - y_batch, axis=0, keepdims=True)  

                #grad_w = np.dot(X_batch.T, y_pred - y_batch) / self.batch_size + 2 * l2 * weights
                #grad_b = np.mean(y_pred - y_batch, axis=0, keepdims=True)

                weights -= self.learning_rate * grad_w
                bias -= self.learning_rate * grad_b

            y_pred = np.dot(X_val, weights) + bias
            loss = np.mean(np.square(y_pred - y_val)) + l2 * np.mean(np.square(weights))
            #mse.append(loss)

            #print("Epoch: {}, Loss: {}".format(epoch, loss))

            

            if count_petience_increase > self.patience:
                mse.append(loss)
                break
            else :
                if loss < best_loss:
                    best_loss = loss
                    count_petience_increase = 0
                    mse.append(loss)
                else:
                    count_petience_increase += 1
                    mse.append(loss)



            """
            if loss < best_loss:
                best_loss = loss
                count_petience_increase = 0
                mse.append(loss)
            else:
                count_petience_increase += 1
                #mse.append(loss)
                if count_petience_increase > self.patience:
                    mse.append(loss)
                    break
            """

        self.weights = weights
        self.bias = bias
        self.mse = mse
        self.mean_sqrd_error = list_average = sum(mse) / len(mse)
        


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.square(y_pred - y))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.weights = model.weights
        self.bias = model.bias

