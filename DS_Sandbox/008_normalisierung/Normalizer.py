import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
data = iris.data
labels = iris.target

x, y = data, labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#plot sepal features
plt.scatter(x_test[:, 0], x_test[:, 1])
plt.show()

#plot petal features
plt.scatter(x_test[:, 2], x_test[:, 3])
plt.show()

print(f"Min: {np.min(x_test, axis=0)}")
print(f"Max: {np.max(x_test, axis=0)}")
print(f"Mean: {np.mean(x_test, axis=0)}")
print(f"Std: {np.std(x_test, axis=0)}")

print("----------------------------")

class Normalizer:
    def __init__(self):
        self.data_min: np.ndarray = None #min value
        self.data_max: np.ndarray = None #max value

    def fitRange(self, x: np.ndarray):
        self.data_min = np.min(x, axis=0)
        self.data_max = np.max(x, axis=0)

    def transformData(self, x: np.ndarray):
        x_transformed = (x - self.data_min) / (self.data_max - self.data_min)
        return x_transformed


scaler = Normalizer()
scaler.fitRange(x)
x_train_transformed = scaler.transformData(x_train)
x_test_transformed = scaler.transformData(x_test)

#plot normalized sepal features
plt.scatter(x_test_transformed[:, 0], x_test_transformed[:, 1])
plt.show()

print(f"Min: {np.min(x_test_transformed, axis=0)}")
print(f"Max: {np.max(x_test_transformed, axis=0)}")
print(f"Mean: {np.mean(x_test_transformed, axis=0)}")
print(f"Std: {np.std(x_test_transformed, axis=0)}")

#plot normalized petal features
plt.scatter(x_test_transformed[:, 2], x_test_transformed[:, 3])
plt.show()
