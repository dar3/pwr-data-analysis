import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# variable
a = 1

# list of available variables
print(dir())

# matrix filled with zeros
# 2 rows and 3 columns
print(np.zeros((2,3)))

# matrix filled with ones
print(np.ones((2,3)))

# identity matrix
print("Identity matrix: ")
print(np.eye(3))
B = np.eye(5)

# copying matrixes
print(np.tile(B,(2,3)))