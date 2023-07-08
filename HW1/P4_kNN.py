import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data = sp.io.loadmat('HW1/digits.mat')
data = list(zip(data['X'], data['Y']))

# Function to separate the data into training and testing sets
def split_data(data, train_size):
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Split the data into training and testing sets
train_data, test_data = split_data(data, 5000)