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
data_dict = {}

for i in range(10):
    data_dict[i] = []

for i in range(len(train_data)):
    label = int(train_data[i][1][0])  # Extract the single value from the array and convert to int
    data_dict[label].append(train_data[i][0])

mean_dict = {}
cov_dict = {}

# NOTE: Better algorithm to calculate the covariance matrix is needed

for label, images in data_dict.items():
    # convert the list of images into a 2D numpy array
    images_array = np.array(images)
    
    # calculate the mean and covariance
    mean = np.mean(images_array, axis=0)
    cov = np.cov(images_array, rowvar=False)

    # store the mean and covariance in dictionaries
    mean_dict[label] = mean
    cov_dict[label] = cov

# NOTE: We added a small value to the diagonal of the covariance matrix to ensure that it is positive definite
distributions = {}
regularization_value = 1e-3

for label in data_dict.keys():
    # Create a covariance matrix for this label, adding a small value to the diagonal
    cov = cov_dict[label] + regularization_value * np.eye(cov_dict[label].shape[0])
    
    # Create a distribution for this label
    distributions[label] = multivariate_normal(mean=mean_dict[label], cov=cov)


def classify_image(image, distributions):
    # Initialize a dictionary to store the likelihoods
    likelihoods = {}

    for label, distribution in distributions.items():
        # Compute the likelihood of the image under this distribution
        likelihoods[label] = distribution.logpdf(image)

    # Return the label with the highest likelihood
    return max(likelihoods, key=likelihoods.get)

# Convert labels in test_data from numpy arrays to integers
test_data = [(image, int(label[0])) for image, label in test_data]

predicted_labels = []
true_labels = [label for image, label in test_data]
# Loop over the test data
for image, true_label in test_data:
    # Classify the image
    predicted_label = classify_image(image, distributions)
    
    # Store the predicted label
    predicted_labels.append(predicted_label)

# Calculate the accuracy by comparing the predicted labels to the true labels
accuracy = sum(predicted_label == true_label for predicted_label, true_label in zip(predicted_labels, true_labels)) / len(test_data)

print("The classification accuracy is: ", accuracy)