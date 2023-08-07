import numpy as np

# Function for running Lloyd's algorithm
def lloyd(points, centers):
    # Initialize array for storing the assignment of each point
    assignment = np.zeros(len(points), dtype=int)

    for iteration in range(10):
        # For each point, find the closest center and assign the point to that center
        for i in range(len(points)):
            # Calculate the distance between the point and each center
            distances = np.linalg.norm(points[i] - centers, axis=1)

            # Find the index of the center with the smallest distance
            assignment[i] = np.argmin(distances)

        # For each cluster, calculate the centroid of the points assigned to it
        for i in range(len(centers)):
            # Find the points assigned to the current cluster
            cluster_points = points[assignment == i]

            # Calculate the centroid of the points
            centers[i] = np.mean(cluster_points, axis=0)

    return assignment, centers
