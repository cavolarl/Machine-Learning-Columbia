import numpy as np

# This is a file just containing the lloyd function
def lloyd(points, centers):
    assignment = np.zeros(len(points), dtype=int)

    for iteration in range(10):
        # For each point, find the closest center and assign the point to that center
        for i in range(len(points)):
            distances = np.linalg.norm(points[i] - centers, axis=1)
            assignment[i] = np.argmin(distances)

        # For each cluster, calculate the centroid of the points assigned to it then update the center to be the centroid
        for i in range(len(centers)):
            cluster_points = points[assignment == i]
            centers[i] = np.mean(cluster_points, axis=0)

    return assignment, centers
