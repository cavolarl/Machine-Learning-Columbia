import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Create a 2x2 matrix L with values 5/4, -3/2, -3/2, 5 and print it
L = np.array([[5/4, -3/2], [-3/2, 5]])
print(L)

def createVector(n):
    # Generate a random n dimensional unit vector v, using the gaussian distribution N(0,1) a.k.a. the standard normal distribution
    v = np.random.normal(0, 1, n)
    # Normalise v
    v = v / np.linalg.norm(v)
    return v

# create 500 random unit vectors of dimension 2 and save them to R
R = np.array([createVector(2) for i in range(500)])

# multiply each vector in R by L and save the result to R2
R2 = np.array([L.dot(v) for v in R])


# compute the eigenvalues of L and save the maximum eigenvalue to max_eigval and the minimum eigenvalue to min_eigval
eigvals = np.linalg.eigvals(L)
max_eigval = np.max(eigvals)
min_eigval = np.min(eigvals)
print(max_eigval)
print(min_eigval)

# for each vector in R2, compute the length
lengths = np.array([np.linalg.norm(v) for v in R2])


# Create a histogram of the lengths of the vectors in R2 using 50 bins
plt.hist(lengths, bins=50)
plt.show()

# NOTE: The histogram has it's maximum at the maximum eigenvalue of L and it's minimum at the minimum eigenvalue of L
# NOTE: The length of a vector v after multiplying it by L is equal to the length of v multiplied by the maximum eigenvalue of L
# NOTE: The minimum eigenvalue of L is the smallest possible value that the length of a vector v after multiplying it by L can have

# Compute the eigenvectors of L and save them to eigvecs
eigvecs = np.linalg.eig(L)[1]
# Then print the maximmum eigenvector
print(eigvecs[:,0])

# NOTE: The maximum eigenvector of L is the vector that when multiplied by L gives the maximum eigenvalue of L multiplied by itself

# plot the vectors in R2 and the maximum eigenvector of L, using a line plot
# set the x and y axis to have the same scale
plt.axis('equal')
# plot the vectors in R2
plt.plot(R2[:,0], R2[:,1], 'o-')
# plot the maximum eigenvector of L
plt.plot([0, eigvecs[0,0]], [0, eigvecs[1,0]], 'r-')
plt.show()



