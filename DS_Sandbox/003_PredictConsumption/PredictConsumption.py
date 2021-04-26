def mpg_to_l_per_100km(mpg):
    LITERS_PER_GALLON = 3.7851411784
    KILOMETERS_PER_MILES = 1.609344

    return (100 * LITERS_PER_GALLON) / (KILOMETERS_PER_MILES * mpg)

print(mpg_to_l_per_100km(100))

import pandas as pd
df = pd.read_csv("mpg-dataset.csv")
#print(df)

X = df[["cylinders", "horsepower", "weight"]]
print(X)
y = mpg_to_l_per_100km(df["mpg"])
#print(y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

print(model.coef_)
print(model.intercept_)

print(model.predict([[8,130,3504],[8,165,3693]]))