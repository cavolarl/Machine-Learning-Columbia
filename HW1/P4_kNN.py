import numpy as np
from collections import Counter
import scipy as sp

# Load the data
data = sp.io.loadmat('digits.mat')
X = data['X']
Y = data['Y'].flatten()

# Split the data into training and test sets
X_train = X[:8000]
y_train = Y[:8000]
X_test = X[8000:]
y_test = Y[8000:]

X_train = X_train / 255
X_test = X_test / 255

# NOTE: Rewrite distance function to manually compute Euclidean distance
# Using reshape to convert 1D array to 2D array
# Using flatten to convert 2D array to 1D array
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
        # NOTE: Using scipy gives an accuracy of 0.955, and using the manual gives an accuracy of 0.232
        # Compute Euclidean distances
        #distances = sp.spatial.distance.cdist(self.X_train, x.reshape(1,-1), 'euclidean').flatten()
        distances = euclidean_distance(x, self.X_train)
        # Get indices of k nearest neighbours
        k_indices = distances.argsort()[:self.k]
        # Get the labels of the k nearest neighbours
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Test the model
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("The classification accuracy is: ", accuracy)
