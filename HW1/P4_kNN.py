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

# Split data, also randomizes the data
def split_data(X, y, split_ratio):
    # Generate a permutation of the indices
    indices = np.random.permutation(X.shape[0])
    split_index = int(split_ratio * X.shape[0])
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test


# Distance function
def euclidean_distance(x1, X):
    return np.sqrt(np.sum((x1 - X)**2, axis=1))

def manhattan_distance(x1, X):
    return np.sum(np.abs(x1 - X), axis=1)

def chebyshev_distance(x1, X):
    return np.max(np.abs(x1 - X), axis=1)

# This class contains the functions of the knn algorithm
class KNN:
    def __init__(self, k, L):
        self.k = k
        self.distance_function = L

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
        distances = self.distance_function(x, self.X_train)
        # sort distance and get the k nearest neighbours
        k_indices = distances.argsort()[:self.k]
        # Get the labels of those k nearest neighbours
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label among them
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Test the model, beware of very low values, they can cause missing labels
split_ratio = 0.7
# Write the distance function you want to use here
distance_function = chebyshev_distance
X_train, y_train, X_test, y_test = split_data(X, y, split_ratio)
# Change k if you want it to be different
knn = KNN(k=7, L=distance_function)
knn.fit(X_train, y_train)
predictions = knn.predict_labels(X_test)
accuracy = np.mean(predictions == y_test)
print(f"The classification accuracy with distance function {str(distance_function)} at a split ratio of: {split_ratio} is: {accuracy}")
