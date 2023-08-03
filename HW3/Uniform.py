import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n_samples = 10000

# Dimensions to test
dimensions = [1, 2, 3, 5, 10, 50, 100]

plt.figure(figsize=(12, 8))

for d in dimensions:
    # Sample uniformly from [-1, 1]^d
    samples = np.random.uniform(-1, 1, (n_samples, d))
    
    # Compute the squared length of each sample
    squared_lengths = np.sum(samples**2, axis=1)
    
    # Plot histogram of squared lengths
    plt.hist(squared_lengths, bins=50, alpha=0.5, label=f'd={d}', density=True)

plt.title("Histogram of Squared Lengths of Samples from Uniform Distributions")
plt.xlabel("Squared Length")
plt.ylabel("Density")
plt.legend()
plt.show()
