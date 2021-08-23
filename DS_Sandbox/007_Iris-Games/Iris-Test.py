import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("iris.data")
print(df[50:51])

#Auswahl von setosa und versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'iris-setosa', -1, 1)

#Auswahl von Kelch- und Blütenlänge
X = df.iloc[0:100, [0,2]].values

#Diagramm plotten
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100,1], color='blue', marker='x', label='versicolor')

plt.xlabel('Länge des Kelchblatts [cm]')
plt.ylabel('Länge des Blüttenblatts [cm]')
plt.legend(loc='upper left')
plt.show()

