from sklearn.linear_model import LinearRegression

X = [[0],[-10],[25]]
y = [[32],[14],[77]]

model = LinearRegression(fit_intercept = True)#Enable Bias
model.fit(X,y)

print(model.coef_)
print(model.intercept_)#Versatz

#Formel = x1 * 1.8 + 32