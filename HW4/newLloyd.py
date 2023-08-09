import numpy as np
from scipy.spatial import distance
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

from datasets import ConcentricCirclesDataset, SparseDataset, ChainDataset

def constructEdgeMatrix(data, r):
    n = len(data)
    matrix = np.zeros((n, n))
    for i in range(n):
        distances = np.linalg.norm(data - data[i], axis=1)
        idx = np.argsort(distances)
        for j in idx[1:r+1]:  # exclude the point itself
            matrix[i, j] = 1
            matrix[j, i] = 1
    return matrix

def computeLaplacian(W):
    D = np.diag(np.sum(W, axis=1))
    return D - W

def bottomKEigenvectors(L, k):
    vals, vecs = eigs(L, k=k, which="SM")
    return vecs

# This is inspired by the GeeksforGeeks page on kmeans++ initialization
def kmeansPlusPlusInit(data, k):
    centers = [data[np.random.randint(data.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centers]) for x in data])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centers.append(data[i])
    return np.array(centers)

def lloydsAlgorithm(dataPoints, centers):
    assignment = np.zeros(len(dataPoints), dtype=int)

    for iteration in range(10):
        for i in range(len(dataPoints)):
            distances = np.linalg.norm(dataPoints[i] - centers, axis=1)
            assignment[i] = np.argmin(distances)

        for i in range(len(centers)):
            clusterPoints = dataPoints[assignment == i]
            if len(clusterPoints) > 0:
                centers[i] = np.mean(clusterPoints, axis=0)

    return assignment, centers

def transformativeClustering(data, r, k):
    edgeMatrix = constructEdgeMatrix(data, r)
    laplacian = computeLaplacian(edgeMatrix)
    eigVectors = bottomKEigenvectors(laplacian, k)
    
    initialCentroids = kmeansPlusPlusInit(eigVectors.real, k)
    assignment, _ = lloydsAlgorithm(eigVectors.real, initialCentroids)

    return assignment


# Create a dataset instance, update this to use the dataset you want to test
dataset = ChainDataset()
dataPoints = dataset.points

# Set parameters
numNeighbors = 10
numClusters = 5

labels = transformativeClustering(dataPoints, numNeighbors, numClusters)

plt.scatter(dataPoints[:, 0], dataPoints[:, 1], c=labels)
plt.title(f"Transformative Clustering on {dataset.name}")
plt.show()