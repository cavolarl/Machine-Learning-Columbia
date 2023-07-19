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
n, d = X_train.shape  # n = number of examples, d = number of features

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

def train_perceptron(X_train, y_train, T):
    w = np.zeros(d)
    w_list = [w]
    c = [0]
    k = 0

    for t in range(T):
        i = t % n
        x_i = X_train[i]
        y_i = y_train[i]
        if y_i * np.dot(w, x_i) <= 0:
            w = w + y_i * x_i
            w_list.append(w)
            c.append(1)
            k += 1
        else:
            c[k] += 1

    return w_list, c

num_classes = 10
weights = []
counts = []

for c in range(num_classes):
    y_train_c = np.where(y_train == c, 1, -1)
    w_list_c, c_list_c = train_perceptron(X_train, y_train_c, n*5)
    weights.append(w_list_c)
    counts.append(c_list_c)
    print(f"Trained perceptron for class {c}")


def predict(x, w_list, c_list):
    score = sum(c_list[i] * np.sign(np.dot(w_list[i], x)) for i in range(len(w_list)))
    return score

def predict_all(X_test):
    scores = []
    for x in X_test:
        class_scores = [predict(x, weights[c], counts[c]) for c in range(num_classes)]
        scores.append(class_scores)
    return np.argmax(scores, axis=1)

# Evaluate on test set
predictions = predict_all(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

