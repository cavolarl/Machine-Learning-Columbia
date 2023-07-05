import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# load the data
data = sp.io.loadmat("HW0/hw0data.mat")

# print the dimensions of the M matrix
print(data['M'].shape)

# print the 4th row and 5th columnn entry of M
# we count from 0, so the 4th row is row 3 and the 5th column is column 4
print(data['M'][3,4])

# print the mean value of the 5th column of M
print(np.mean(data['M'][:,4]))

# compute a histogram of the 4th row of M and plot it
plt.hist(data['M'][3,:])
plt.show()

# compute and print the top three eigenvalues of MtM
# transposing M with .T and multiplying it with M gives us MtM (Using .dot to multiply)
eigvals = np.linalg.eigvals(data['M'].T.dot(data['M']))
print(eigvals[0:3])
