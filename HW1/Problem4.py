import scipy as sp
import numpy as np

data = sp.io.loadmat('HW1/digits.mat')

# Our data consists of 10000 28x28 images of handwritten digits

# X is a 10000x784 matrix where each row is a represetentation of the handwritten digits
# Y is a 10000x1 matrix of the labels for each image

X = data['X']
Y = data['Y']

# We will use the first 8000 images as our training set and the last 2000 as our test set

X_train = X[:8000]
Y_train = Y[:8000]

X_test = X[8000:]
Y_test = Y[8000:]

# We create tuples of (image, label) for each image in our training set
# NOTE: I don't know if we need tuples but we proceed like this for now

training_set = [(X_train[i], Y_train[i]) for i in range(len(X_train))]
test_set = [(X_test[i], Y_test[i]) for i in range(len(X_test))]

# NOTE: We are going to create two different classifiers, one using a MLE to evaluate the density function, the other using kNN
# We will then compare the two classifiers to see which one performs better

# Using MLE to evaluate the density function
# TODO: Implement this

# Using kNN
# TODO: Implement this

# TODO: Compare the two classifiers

