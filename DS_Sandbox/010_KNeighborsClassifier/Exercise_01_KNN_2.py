import numpy as np
np.random.seed(42)
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
labels = iris.target
names = iris.target_names
features = iris.values()
from scipy.spatial import distance

#test_data = [5.2,4.1,1.5,0.1]      #0
#test_data = [5.2,2.7,3.9,1.4]      #1
test_data = [4.8, 2.5, 5.3, 2.4]   #2

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

class KNeighborsClassifier:
    def __init__(self, k: int, data: np.ndarray, test_data: list):
        self.k = k
        self.data = data
        self.test_data = test_data

    def normalizer(self):
        x = self.data
        data_min = np.min(x, axis=0)
        data_max = np.max(x, axis=0)
        x_transformed = (x - data_min) / (data_max - data_min)
        return x_transformed

    def distance(self):
        distances = list()
        for i in self.data:
            distances.append(distance.euclidean(i, self.test_data))
        return distances

    def sortDistances(self, x:list):
        return np.argsort(x)

    def determineNeighbors(self, distances: np.ndarray, y: np.ndarray):
        targetList = list()
        neighbors = distances[0:self.k]
        for i in neighbors:
            print("label ", self.data[i])
            targetList.append(y[i])
        return targetList

scaler = Normalizer()
scaler.fitRange(data)
transformed_data = scaler.transformData(data)
print("transformed data\n", transformed_data[:10])

clf = KNeighborsClassifier(12, data, test_data)

normalized_data = clf.normalizer()
print("normalized data\n", normalized_data[:10])

distances = clf.distance()
print("distances ", distances[:10])

sorted_distances = clf.sortDistances(distances)
print("sorted distances ", sorted_distances[:10])

nNeighbors = clf.determineNeighbors(sorted_distances, data)
print("nNeighbors coordinates ", nNeighbors)

nNeighbors = clf.determineNeighbors(sorted_distances, labels)
print("nNeighbors classes ", nNeighbors)

for name in nNeighbors:
    print("iris ", names[name])
