import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy as np

# loadding data using pandas

df = pd.read_csv("FuelConsumptionCo2.csv") 

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='red')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

# Training and test data-set

msk = np.random.rand(len(cdf)) < 0.7

train = cdf[msk]
test = cdf[~msk]

linr_model = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
linr_model.fit(train_x,train_y)

print("Intercept : ",linr_model.intercept_)
print("coeffs :", linr_model.coef_)

y_hat = linr_model.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])

print("Predicted : ",y_hat)

test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f"
      % np.mean((y_hat - test_y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linr_model.score(test_x, test_y))