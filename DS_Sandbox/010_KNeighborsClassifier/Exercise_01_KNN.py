__author__ = "Salim Oussayfi"

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial import distance
iris = load_iris()
data = iris.data[:10]
labels = iris.target
names = iris.target_names

print("names ", names[:10])
print("labels",labels[:10])


#test_data = [5.2,4.1,1.5,0.1]      #0
#test_data = [5.2,2.7,3.9,1.4]      #1
test_data = [4.8, 2.5, 5.3, 2.4]   #2

class KNeighborsClassifier:
    def __init__(self, k: int, data: np.ndarray, names: list, labels: np.ndarray, test_data: list):
        self.k = k
        self.data = data
        self.names = names
        self.labels = labels
        self.test_data = test_data

        normalized_data = self.normalizeData()
        distances = self.euclideanDistance()
        sorted_distances = self.sortDistances(distances)
        nNeighbors = self.determineNeighbors(sorted_distances, self.labels)

        print('The {} closest neighbors of searched iris {} are:'.format(self.k, self.test_data))
        for name in nNeighbors:
            print("iris ", self.names[name])

        self.determinePropability(nNeighbors)

    def normalizeData(self):
        x = self.data
        data_min = np.min(x, axis=0)
        data_max = np.max(x, axis=0)
        x_transformed = (x - data_min) / (data_max - data_min)
        return x_transformed

    def euclideanDistance(self):
        distances = list()
        for i in self.data:
            distances.append(distance.euclidean(i, self.test_data))
        return distances

    def sortDistances(self, x:list):
        print("neighbors ",np.argsort(x))
        return np.argsort(x)

    def determineNeighbors(self, distances: np.ndarray, y: np.ndarray):
        targetList = list()
        neighbors = distances[0:self.k]
        for i in neighbors:
            targetList.append(y[i])
        return targetList

    def determinePropability(self, nNeighbors:list):
        amount = len(nNeighbors)
        closedNeighbor = nNeighbors[0]
        occurrences = nNeighbors.count(closedNeighbor)
        propability = occurrences/amount
        print('Propability for iris {} by examine next {} neighbors is {} %'.format(self.names[closedNeighbor], self.k, propability*100))

#clf = KNeighborsClassifier(8, data, names, labels, test_data)

movies_ = pd.read_csv("movies.csv")
movies = np.array(movies_)
data2 = movies[:,[0,1]]
# print(data2)
names2 = np.unique(movies[:,2])
print("names ", names2)
labels2 = movies[:,3]
print("labels", labels2)

test_data2 = [5.0, 140.0]

clf1 = KNeighborsClassifier(5, data2, names2, labels2, test_data2)

test_data3 = [105.0, 4.0]
#clf2 = KNeighborsClassifier(5, data2, names2, labels2, test_data3)




