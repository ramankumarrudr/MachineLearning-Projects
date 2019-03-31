import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
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

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)				#polynomialFeatures() is used to extend the features of independent variable
train_x_poly = poly.fit_transform(train_x)

linr_model = linear_model.LinearRegression()

train_y_out = linr_model.fit(train_x_poly,train_y)

print("Intercept : ",linr_model.intercept_)
print("coeffs :", linr_model.coef_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)	#aranges (0.0,0.1,0.2,0.3 _________ ,10.0)
yy = linr_model.intercept_[0]+ linr_model.coef_[0][1]*XX+ linr_model.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_ = linr_model.predict(test_x_poly)

# Calculating the R2 score and MSE
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
