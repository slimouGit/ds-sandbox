# - funct: You normalize your data in another table
# - funct: You code a simple euclid distance function
# - funct: You take a point and calculate the distance to all points
# - funct: You take the list from above and sort it
# - funct: You aggregate by target variable
# - funct: you take the max to determine the targe class

import numpy as np
np.random.seed(42)
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
labels = iris.target
features = iris.values()
from scipy.spatial import distance

#exerciseData = [5.2,4.1,1.5,0.1]
exerciseData = [5.2,2.7,3.9,1.4]
#exerciseData = [4.8, 2.5, 5.3, 2.4]

class KNeighborsClassifier:
    def __init__(self, k: int = 5):
        self.k = k

    def normalizer(X: np.ndarray):
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        x_transformed = (X - data_min) / (data_max - data_min)
        return x_transformed

    def distance(x:np.ndarray, y):
        distances = list()
        for i in x:
            distances.append(distance.euclidean(i, y))
        return distances

    def sortDistances(x:list):
        return np.argsort(x)

    def predictTargetClass(x: list, y: np.ndarray):
        targetList = list()
        neighbors = x[0:3]
        for i in neighbors:
            targetList.append(y[i])
        return targetList

    def predictTargetClass2(x:list, y: np.ndarray):
        targetList= list()
        neighbors = x[0:3]
        for i in neighbors:
            targetList.append(y[i])
        return targetList


clf = KNeighborsClassifier

x_transformed = clf.normalizer(data)
print("x ",data[:3])
print("x_transformed ",x_transformed[:3])
print("exerciseData ",exerciseData)

distances = clf.distance(data,exerciseData)
print("distances ", distances[:3])

sortedDistances = clf.sortDistances(distances)
print("sortedDistances ", sortedDistances[:3])

targetClasses = clf.predictTargetClass(sortedDistances, data)
# print(targetClasses)

targetClasses2 = clf.predictTargetClass2(sortedDistances, labels)
print("Classes: ", targetClasses2)





