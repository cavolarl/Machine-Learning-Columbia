import scipy as sp
import numpy as np

### Written by Carl Lavo
### Uni: cgl2131
### E-mail: cgl2131@columbia.edu
### COMS 4771 Machine Learning

# Load the data
data = sp.io.loadmat('digits.mat')
X = data['X'] / 255   # Normalize the features
y = data['Y'].flatten()

# Split the data into training and test sets
# Im using a split of 70/30, I found it to work pretty well.
split_ratio = 0.7
split_index = int(split_ratio * X.shape[0])
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]


# This is our definition of a multivariate normal distribution
class MultivariateNormal:
    def __init__(self, mean, cov):
        self.dim = mean.shape[0]
        self.mean = mean
        # We add a small constant to the cov, to ensure stability.
        self.cov = cov + 1e-6 * np.eye(self.dim)
        self.inv_cov = np.linalg.inv(self.cov)
        # Using this slogdet, improves the stability of the calculation
        self.log_det_cov = np.linalg.slogdet(self.cov)[1]
    # probability density function
    def pdf(self, x):
        x_m = x - self.mean
        return np.exp(-0.5 * (x_m @ self.inv_cov @ x_m)) / np.sqrt((2 * np.pi) ** self.dim * np.exp(self.log_det_cov))
    # log probability density function
    def logpdf(self, x):
        x_m = x - self.mean
        return -0.5 * (x_m @ self.inv_cov @ x_m) - 0.5 * self.dim * np.log(2 * np.pi) - 0.5 * self.log_det_cov

# Calculate the mean and covariance for each label
mean_dict = {}
cov_dict = {}
for label in range(10):
    X_label = X_train[y_train == label]
    mean_dict[label] = np.mean(X_label, axis=0)
    cov_dict[label] = np.cov(X_label, rowvar=False)

# Create a dictionary to store the distributions for each class
distributions = {}
for label in range(10):
    cov = cov_dict[label] + np.eye(cov_dict[label].shape[0])
    distributions[label] = MultivariateNormal(mean=mean_dict[label], cov=cov)

# Classify the test set
predictions = []
for image in X_test:
    # Calculate the likelihood of the image under each distribution
    likelihoods = [distributions[label].logpdf(image) for label in range(10)]
    # Predict the label with the highest likelihood
    predictions.append(np.argmax(likelihoods))

# Calculate the accuracy
accuracy = np.mean(predictions == y_test)
print(f"The classification accuracy is: {accuracy}")