import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# NOTE: This code is really bad, and I'm not proud of it. I'm just trying to get it to work.

data = sp.io.loadmat('HW1/digits.mat')

# I'm going to create tuples of (x,y) to more easily access the labels

data = list(zip(data['X'], data['Y']))

# We separate the data into training and testing sets

train_data = data[:5000]
test_data = data[5000:]

print(train_data[0][0], train_data[0][1])

print(train_data[0])

# The value in data[0][0] is the image data, and the value in data[0][1] is the label

# I'm going to create a dictionary of lists, where the key is the label and the value is a list of all the images with that label

data_dict = {}

for i in range(10):
    data_dict[i] = []

for i in range(len(train_data)):
    label = int(train_data[i][1][0])  # Extract the single value from the array and convert to int
    data_dict[label].append(train_data[i][0])

# print an example of the data for each label
for i in range(10):
    print("Label: ", i)
    print(data_dict[i][0])

# Now we need to calculate the mean and covariance matrix for each label

mean_dict = {}
cov_dict = {}

for label, images in data_dict.items():
    # convert the list of images into a 2D numpy array
    images_array = np.array(images)
    
    # calculate the mean and covariance
    mean = np.mean(images_array, axis=0)
    cov = np.cov(images_array, rowvar=False)

    # store the mean and covariance in dictionaries
    mean_dict[label] = mean
    cov_dict[label] = cov

# print an example of the mean and covariance for each label
for i in range(10):
    print("Label: ", i)
    print("Mean: ", mean_dict[i])
    print("Covariance: ", cov_dict[i])

from scipy.stats import multivariate_normal

# Initialize a dictionary to store the distributions
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

