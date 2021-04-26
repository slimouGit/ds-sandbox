X = [[50],[60],[70],[20],[10],[30]]
y = [[1],[1],[1],[0],[0],[0]]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 100000)
model.fit(X,y)

print(model.predict([[35]]))
print(model.predict([[65]]))
print(model.predict([[50]]))

print(model.predict_proba([[60]]))
