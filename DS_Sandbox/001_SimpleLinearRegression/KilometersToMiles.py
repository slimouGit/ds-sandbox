from sklearn.linear_model import LinearRegression

X = [[10],[15],[60]]
y = [[6.21371],[9.32057],[37.2823]]

model = LinearRegression(fit_intercept = False)
model.fit(X,y)

print(120 * 0.621371)

prediction = model.predict([[120]])

print(prediction)


