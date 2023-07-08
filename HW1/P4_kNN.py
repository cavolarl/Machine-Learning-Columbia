import scipy as sp
import numpy as np
from collections import Counter

data = sp.io.loadmat('HW1/digits.mat')
data = list(zip(data['X'], data['Y']))

# Function to separate the data into training and testing sets
def split_data(data, train_size):
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Split the data into training and testing sets
train_data, test_data = split_data(data, 8000)

# Separate the images and labels in the training and testing sets
X_train = np.array([x[0] for x in train_data])
y_train = np.array([x[1][0] for x in train_data])  # get the label from each tuple and unwrap it from its array
X_test = np.array([x[0] for x in test_data])
y_test = np.array([x[1][0] for x in test_data])  # get the label from each tuple and unwrap it from its array

# Our distance function (Euclidean distance)
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.y_train = Y

    def predict(self, X):
        predicted_labels = [self.compute_kNN(x) for x in X]
        return np.array(predicted_labels)

    def compute_kNN(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# TODO: Implement a better way of dealing with parameters


knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# To compute accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print("The classification accuracy is: ", accuracy)
