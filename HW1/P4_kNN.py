import numpy as np
from collections import Counter
import scipy as sp

### Written by Carl Lavo
### Uni: cgl2131
### E-mail: cgl2131@columbia.edu
### COMS 4771 Machine Learning

# Load the data
data = sp.io.loadmat('digits.mat')
X = data['X'] / 255   # Normalize the features
y = data['Y'].flatten()

# Split the data into training and test sets
split_ratio = 0.7
split_index = int(split_ratio * X.shape[0])
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Distance function
def euclidean_distance(x1, X):
    return np.sqrt(np.sum((x1 - X)**2, axis=1))

# This class contains the functions of the knn algorithm
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_labels(self, X):
        predicted_labels = np.empty(X.shape[0])
        # Loop over all observations
        for i, observation in enumerate(X):
            predicted_labels[i] = self.predict(observation)
        return predicted_labels

    def predict(self, x):
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
predictions = knn.predict_labels(X_test)
accuracy = np.mean(predictions == y_test)
print(f"The classification accuracy is: {accuracy}")
