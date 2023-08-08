import numpy as np

class ConcentricCirclesDataset:
    def __init__(self, nPoints=200, noise=0.05, radius=[1, 2]):
        self.nPoints = nPoints
        self.noise = noise
        self.radius = radius
        self.points = self.generate()
        self.name = "Concentric Circles"

    def generate(self):
        points = np.zeros((self.nPoints * len(self.radius), 2))

        for i, r in enumerate(self.radius):
            # Generate angles between 0 and 2π
            angles = np.linspace(0, 2 * np.pi, self.nPoints)

            # Generate the points on the circle
            x = r * np.cos(angles) + np.random.normal(0, self.noise, self.nPoints)
            y = r * np.sin(angles) + np.random.normal(0, self.noise, self.nPoints)

            # Store the points in the dataset
            points[i * self.nPoints : (i + 1) * self.nPoints, 0] = x
            points[i * self.nPoints : (i + 1) * self.nPoints, 1] = y

        return points

class SparseDataset:
    def __init__(self, nPointsDense=800, nPointsSparse=200, noiseDense=0.1, noiseSparse=0.5):
        self.nPointsDense = nPointsDense
        self.nPointsSparse = nPointsSparse
        self.noiseDense = noiseDense
        self.noiseSparse = noiseSparse
        self.denseCenters = [(-2, -2), (2, -2)]
        self.sparseCenters = [(-2, 2), (2, 2)]
        self.points = self.generate()
        self.name = "Sparse Dataset"

    def generate(self):
        points = np.zeros((self.nPointsDense + self.nPointsSparse, 2))

        for i, (cx, cy) in enumerate(self.denseCenters):
            x = np.random.normal(cx, self.noiseDense, self.nPointsDense // len(self.denseCenters))
            y = np.random.normal(cy, self.noiseDense, self.nPointsDense // len(self.denseCenters))
            points[i * (self.nPointsDense // len(self.denseCenters)):(i + 1) * (self.nPointsDense // len(self.denseCenters)), 0] = x
            points[i * (self.nPointsDense // len(self.denseCenters)):(i + 1) * (self.nPointsDense // len(self.denseCenters)), 1] = y

        for i, (cx, cy) in enumerate(self.sparseCenters):
            x = np.random.normal(cx, self.noiseSparse, self.nPointsSparse // len(self.sparseCenters))
            y = np.random.normal(cy, self.noiseSparse, self.nPointsSparse // len(self.sparseCenters))
            startIdx = self.nPointsDense + i * (self.nPointsSparse // len(self.sparseCenters))
            endIdx = self.nPointsDense + (i + 1) * (self.nPointsSparse // len(self.sparseCenters))
            points[startIdx:endIdx, 0] = x
            points[startIdx:endIdx, 1] = y

        return points

class ChainOfCirclesDataset:
    def __init__(self, nCircles=5, pointsPerCircle=100, noise=0.05, radiusMultiplier=2):
        self.nCircles = nCircles
        self.pointsPerCircle = pointsPerCircle
        self.noise = noise
        self.radiusMultiplier = radiusMultiplier
        self.points = self.generate()
        self.name = "Chain of Circles Dataset"

    def generate(self):
        points = np.zeros((self.nCircles * self.pointsPerCircle, 2))
        
        # Starting center of the circle chain
        center_x, center_y = 0, 0
        current_radius = self.radiusMultiplier  # Initial circle radius
        
        for i in range(self.nCircles):
            # Generate angles between 0 and 2π for the current circle
            angles = np.linspace(0, 2 * np.pi, self.pointsPerCircle)
            
            # Generate the points on the circle
            x = current_radius * np.cos(angles) + np.random.normal(0, self.noise, self.pointsPerCircle) + center_x
            y = current_radius * np.sin(angles) + np.random.normal(0, self.noise, self.pointsPerCircle) + center_y

            # Store the points in the dataset
            points[i * self.pointsPerCircle : (i + 1) * self.pointsPerCircle, 0] = x
            points[i * self.pointsPerCircle : (i + 1) * self.pointsPerCircle, 1] = y
            
            # Update the center for the next circle. Since we want a chain, we'll set the new center at the end of the current circle
            angle_for_next_center = angles[-1]
            center_x = x[-1] + (current_radius + current_radius * self.radiusMultiplier) * np.cos(angle_for_next_center)
            center_y = y[-1] + (current_radius + current_radius * self.radiusMultiplier) * np.sin(angle_for_next_center)
            
            # Update the radius for the next circle
            current_radius *= self.radiusMultiplier

        return points