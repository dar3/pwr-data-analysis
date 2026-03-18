import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# variable
# a = 1

# list of available variables
# print(dir())

# matrix filled with zeros
# 2 rows and 3 columns
# print(np.zeros((2,3)))

# matrix filled with ones
# print(np.ones((2,3)))

# identity matrix (macierz jednostkowa)
# print("Identity matrix: ")
# print(np.eye(3))
# B = np.eye(5)

# copying matrixes
# print()
# print(np.tile(B,(2,3)))

# macierz rozkladu plaskiego (flat distribution)
# print("Rozklad plaski")
# print(np.random.rand(10, 1))

# macierz rozkladu normalnego (normal distribution)
# print("Rozklad normalny")
# print(np.random.randn(10, 1))

# matrix size (matrix name.shape)
# print(B.shape)



# Exercise 2 histogram vs boxplot

# srednia 1, odchylenie 1
# 100 rows 1 column
x1 = 2 * (np.random.randn(100, 1) + 1)
# srednia -1, odchylenie 1
x2 = 3 * (np.random.randn(100, 1) - 1)

# joining two columns 100 x 2 matrix
z = np.hstack([x1, x2])

plt.figure(figsize=(10, 6))

# Boxplot
plt.subplot(2, 1, 1)
plt.boxplot(z)
plt.title("Boxplot with z data")

# Histogram
plt.subplot(2, 1, 2)
plt.hist(z, label=['x1', 'x2'])
plt.title("Histogram with z data")
plt.legend()
plt.show()

# Exercise 3

data = np.random.randn(300)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(data)
plt.title("Line graph (plot)")

plt.subplot(3, 2, 3)
plt.hist(data, bins=20, color='hotpink', edgecolor='black')
plt.title("Histogram 20 bars")

plt.subplot(3, 2, 4)
plt.hist(data, bins=100, color='tomato', edgecolor='black')
plt.title("Histogram 100 bars")


plt.subplot(3, 1, 3)
plt.boxplot(data, vert=False) # vert=False for clarity
plt.title("Data boxplot")

plt.tight_layout()
plt.show()
