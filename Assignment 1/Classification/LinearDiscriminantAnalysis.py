import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.classes = None
        self.class_means = None
        self.class_covariance = None
        self.class_priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.class_means = np.zeros((num_classes, num_features))
        self.class_covariance = np.zeros((num_features, num_features))
        self.class_priors = np.zeros(num_classes)

        for i, cls in enumerate(self.classes):
            X_c = X[y == cls]
            self.class_means[i] = np.mean(X_c, axis=0)
            self.class_covariance += np.cov(X_c.T, bias=True) * (len(X_c) - 1)
            self.class_priors[i] = len(X_c) / len(X)

        self.class_covariance /= len(X)

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            posteriors = np.zeros(num_classes)

            for j in range(num_classes):
                mean_diff = X[i] - self.class_means[j]
                inv_covariance = np.linalg.inv(self.class_covariance)
                posterior = np.log(self.class_priors[j]) - 0.5 * np.dot(mean_diff, np.dot(inv_covariance, mean_diff.T))
                posteriors[j] = posterior

            predictions[i] = np.argmax(posteriors)

        return predictions
