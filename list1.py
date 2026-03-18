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

# mean 1, odchylenie 1
# 100 rows 1 column
x1 = 2 * (np.random.randn(100, 1) + 1)
# mean -1, odchylenie 1
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
plt.title("Line graph")

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


# Exercise 4

import numpy as np
import matplotlib.pyplot as plt


def gen1(x, N):
    m, a, c = 8191, 101, 1731
    y = np.zeros(N)
    for i in range(N):
        x = (a * x + c) % m
        y[i] = x / m
    return y


def gen2(x, N):
    a, m, c = 517, 32767, 6923
    y = np.zeros(N)
    for i in range(N):
        x = (a * x + c) % m
        y[i] = x / m
    return y


def gen3(x, N):
    c = 65536
    y = np.zeros(N)
    for i in range(N):
        x = (x * 25) % c
        x = (x * 125) % c
        y[i] = x / c
    return y


seed = 1
# samples number
N = 1100

data_randn = np.random.randn(N)
data_gen1 = gen1(seed, N)
data_gen2 = gen2(seed, N)
data_gen3 = gen3(seed, N)

generators = [data_randn, data_gen1, data_gen2, data_gen3]
titles = ["(randn)", "Generator  1", "Generator 2", "Generator 3"]

plt.figure(figsize=(12, 8))

for i, data in enumerate(generators):
    mean = np.mean(data)
    variance = np.var(data)

    print(f"{titles[i]}: Średnia = {mean:.4f}, Wariancja = {variance:.4f}")

    plt.subplot(2, 2, i + 1)
    plt.hist(data, bins= 32, color='khaki', edgecolor='white')
    plt.title(f"{titles[i]}\nMean: {mean:.2f}, Var: {variance:.2f}")

plt.tight_layout()
plt.show()

# Exercise 5

try:
    iris_data = np.genfromtxt('data/iris.txt', skip_header=1, usecols=(1, 2, 3, 4))
    glass_data = np.genfromtxt('data/glass.txt', skip_header=1, usecols=range(0, 9))


    iris_attr = iris_data[:, 0]
    glass_attr = glass_data[:, 1]

    lp = 15

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(iris_attr, bins=lp, color='darkorange', edgecolor='black')
    plt.title(f"Iris - attribute 1 (bins={lp})")
    plt.xlabel("attribute value")


    plt.subplot(1, 2, 2)
    plt.hist(glass_attr, bins=lp, color='springgreen', edgecolor='black')
    plt.title(f"Glass - attribute 2 (bins={lp})")
    plt.xlabel("attribute value")

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("File was not found")