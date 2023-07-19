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

# Perceptron V2
# w_1 = 0, c_0 = 0, k = 1
# for t = 1,2,...,T
# pick example (x_i, y_i), where i = (t mod n + 1)
# if y_i * w_k * x_i <= 0
#   w_k+1 = w_k + y_i * x_i
#   c_k+1 = 1
#   k = k + 1
# else
#   c_k = c_k + 1

# Classifier
# f(x) = sign(sum_{i=1}^k c_i*sign(w_i * x))

n, d = X_train.shape  # n = number of examples, d = number of features
w = np.zeros(d)
c = [0]
k = 0
T = n * 10  # Assuming 10 passes over the dataset

for t in range(T):
    i = t % n
    x_i = X_train[i]
    y_i = y_train[i]
    if y_i * np.dot(w, x_i) <= 0:
        w = w + y_i * x_i
        c.append(1)
        k += 1
    else:
        c[k] += 1



def sign(val):
    return np.sign(val)

def f(x):
    summation = 0
    for i in range(k+1):  # Remember that indexing starts from 0
        summation += c[i] * sign(np.dot(w, x))
    return sign(summation)

def test_classifier(X_test, y_test):
    correct_predictions = 0
    total_predictions = len(y_test)
    
    for i in range(total_predictions):
        prediction = f(X_test[i])
        if prediction == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

accuracy = test_classifier(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

