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

def split_data(X, y, split_ratio):
    indices = np.random.permutation(X.shape[0])
    split_index = int(split_ratio * X.shape[0])
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

# NOTE: Set data split here
X_train, y_train, X_test, y_test = split_data(X, y, 0.7)

# NOTE: Set the kernel function here
def polynomial_kernel(x_i, x_j, d=20, c=5):
    return (c + np.dot(x_i, x_j)) ** d

# NOTE: Also set the kernel function here
def compute_kernel_matrix(X, d=20, c=5):
    K = np.dot(X, X.T)
    K = (c + K) ** d
    return K

K_train = compute_kernel_matrix(X_train)

def train_kernel_perceptron(K_train, y_train, T):
    n_samples = K_train.shape[0]
    alpha = np.zeros(n_samples)
    
    for t in range(T):
        i = t % n_samples
        # This is the misclassification condition
        f_xi = np.sign(np.dot(alpha * y_train, K_train[:, i]))
        if f_xi != y_train[i]:
            alpha[i] += 1
            
    return alpha

def predict_kernel_perceptron(K_test_point, alphas, y_train):
    scores = np.zeros(num_classes)
    for c in range(num_classes):
        score = np.sum(alphas[c] * y_train * K_test_point)
        scores[c] = score
    return np.argmax(scores)


num_classes = 10
alphas = np.zeros((num_classes, X_train.shape[0]))

for c in range(num_classes):
    y_train_c = np.where(y_train == c, 1, -1)
    # NOTE: Set n here
    alphas[c] = train_kernel_perceptron(K_train, y_train_c, len(y_train)*30)
    print(f"Trained kernel perceptron for class {c}")

# Evaluate on test set
predictions = []
for x in X_test:
    K_test_point = polynomial_kernel(X_train, x)
    predictions.append(predict_kernel_perceptron(K_test_point, alphas, y_train))
predictions = np.array(predictions)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy*100:.2f}%")
