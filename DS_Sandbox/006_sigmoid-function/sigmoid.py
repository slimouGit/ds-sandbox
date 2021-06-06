import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'dataset.csv')
print(df.head(10))
x = df.iloc[:,[2,3]].values
y = df.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting logistic regression to the training set
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

# Predicting the test results
y_pred = model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))

