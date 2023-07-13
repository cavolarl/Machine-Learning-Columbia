import scipy as sp
import numpy as np

data = sp.io.loadmat('digits.mat')
data = list(zip(data['X'], data['Y']))

# Define a multivariate normal distribution
class MultivariateNormal:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        self.dim = len(mean)
        
    def pdf(self, x):
        """
        Calculate the Probability Density Function at data point x
        """
        x_m = x - self.mean
        return (1. / (np.sqrt((2 * np.pi)**self.dim * self.det_cov)) *
                np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))

    def logpdf(self, x):
        """
        Calculate the log Probability Density Function at data point x
        """
        x_m = x - self.mean
        return (-0.5 * self.dim * np.log(2 * np.pi) - 0.5 * np.log(self.det_cov) -
                0.5 * x_m.T.dot(self.inv_cov).dot(x_m))


# Function to separate the data into training and testing sets
def split_data(data, train_size):
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Split the data into training and testing sets
train_data, test_data = split_data(data, 7000)
data_dict = {}

for i in range(10):
    data_dict[i] = []

for i in range(len(train_data)):
    label = int(train_data[i][1][0])  # Extract the single value from the array and convert to int
    data_dict[label].append(train_data[i][0])

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

# NOTE: We added a small value to the diagonal of the covariance matrix to ensure that it is positive definite
distributions = {}
regularization_value = 1e-3
for label in data_dict.keys():
    # Create a covariance matrix for this label
    cov = cov_dict[label] + np.eye(cov_dict[label].shape[0]) * regularization_value
    # Create a distribution for this label using a multivariate normal(gaussian distribution)
    distributions[label] = MultivariateNormal(mean=mean_dict[label], cov=cov)


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