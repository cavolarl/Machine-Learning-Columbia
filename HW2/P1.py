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