import numpy as np
from collections import Counter
import scipy as sp

# Load the data
data = sp.io.loadmat('digits.mat')
X = data['X']
# Convert Y from 2D to 1D
Y = data['Y'].flatten()

# Split the data into training and test sets
X_train = X[:8000]
y_train = Y[:8000]
X_test = X[8000:]
y_test = Y[8000:]

# Normalize the data, (this improved the accuracy by 70%)
X_train = X_train / 255
X_test = X_test / 255

# This is our euclidean distance function
# axis = 1 means we want to sum over the rows
def euclidean_distance(x1, X):
    return np.sqrt(np.sum((x1 - X)**2, axis=1))


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = np.empty(X.shape[0])
        # Loop over all observations
        for i, observation in enumerate(X):
            predicted_labels[i] = self._predict(observation)
        return predicted_labels

    def _predict(self, x):
        distances = euclidean_distance(x, self.X_train)
        # sort distance and get the k nearest neighbours
        k_indices = distances.argsort()[:self.k]
        # Get the labels of those k nearest neighbours
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label among them
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Test the model
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("The classification accuracy is: ", accuracy)
