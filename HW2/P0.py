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


# Initialize w_0 as zero
# for t = 1, 2, ..., T:
# Pick an example (x_i, y_i) where i=(t mod n + 1)

# If y_i(w_(t-1) dot x_i) <= 0:
# w_t = w_(t-1) + y_i*x_i
# Else:
# w_t = w_(t-1)

# Classifier
# Sign(w_T dot x)

# NOTE: Set data split here
X_train, y_train, X_test, y_test = split_data(X, y, 0.01)
n, d = X_train.shape

def train_perceptron(X_train, y_train, T):
    w = np.zeros(d)
    for t in range(T):
        i = t % n  # Adjusted for 0-based indexing
        x_i = X_train[i]
        y_i = y_train[i]
        if y_i * np.dot(w, x_i) <= 0:
            w = w + y_i * x_i
    return w

# Create a weight matrix to store weight vectors for each class
num_classes = 10
weights = np.zeros((num_classes, X_train.shape[1]))

# Train a perceptron for each class
# NOTE: Parameters can be set here
for c in range(num_classes):
    y_train_c = np.where(y_train == c, 1, -1)  # Adjust labels for current class
    weights[c] = train_perceptron(X_train, y_train_c, n*5)  # Assuming 10 passes over the dataset for each perceptron

def predict(x):
    # Compute scores for each class
    scores = np.dot(weights, x)
    # Return class with highest score
    return np.argmax(scores)

# Evaluate on test set
predictions = np.array([predict(x) for x in X_test])
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy*100:.2f}%")
