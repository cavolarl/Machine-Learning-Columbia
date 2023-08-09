import numpy as np
import matplotlib.pyplot as plt
from lloyd import lloyd
from datasets import SparseDataset

data = SparseDataset()

points = data.points

# Determine the bounds of the dataset for better initialization
x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

# Then initialize n centers randomly within the bounds, then run lloyds
centers = np.array([[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for _ in range(4)])
assignment, centers = lloyd(points, centers)

# Visualize the points and their clusters
plt.scatter(points[:, 0], points[:, 1], c=assignment)
plt.scatter(centers[:, 0], centers[:, 1], c='red')
plt.show()
