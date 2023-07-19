import numpy as np
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

# Split the data
X_train, y_train, X_test, y_test = split_data(X, y, 0.7)
n, d = X_train.shape

# Perceptron V1

# Initialize w_0 = 0
# for t = 1,2,...,T
# pick example (x_i,y_i) such that i = arg min_j (y_j * w_t-1 * x_j)
# if y_i * w_t-1 * x_i <= 0
#   w_t = w_t-1 + y_i * x_i
# else
#   w_T = w_t-1
# end

# Classifier
# f(x) = sign(w_T * x)

import numpy as np

# This has been changed to utilise matrix multiplication, since my PC wanted to unalive itself
def train_perceptron(X_train, y_train, T):
    w = np.zeros(d)

    for t in range(T):
        # 1. Compute dot products for all data points at once
        predictions = np.dot(X_train, w)
        
        # 2. Compute the indices of all misclassified data points
        misclassified_indices = np.where(y_train * predictions <= 0)[0]
        
        if len(misclassified_indices) == 0:
            # No misclassified samples, we've achieved linear separability
            break
        
        # Use one misclassified example to update the weight
        i = misclassified_indices[0]
        x_i = X_train[i]
        y_i = y_train[i]
        
        # Update weights using the misclassified example
        w += y_i * x_i

    return w


# Create a weight matrix to store weight vectors for each class
num_classes = 10
weights = np.zeros((num_classes, X_train.shape[1]))

# Train a perceptron for each class
for c in range(num_classes):
    y_train_c = np.where(y_train == c, 1, -1)
    weights[c] = train_perceptron(X_train, y_train_c, n*5)
    print(f"Trained perceptron for class {c}")

def predict_all(X_test):
    # Compute scores for each data point for all classes at once
    scores = np.dot(X_test, weights.T)
    # Return classes with the highest scores
    return np.argmax(scores, axis=1)

# Evaluate on test set
predictions = predict_all(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

# NOTE: Running this code takes 5-10 minutes on my PC