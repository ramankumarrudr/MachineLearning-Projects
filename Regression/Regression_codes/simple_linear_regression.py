import matplotlib .pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import r2_score
from sklearn import linear_model

# Reading the Data using pandas

df = pd.read_csv("FuelConsumptionCo2.csv")

print(df.head()) # dis plays the top data

print(df.describe()) #discriptive exploration

class_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] #selecting the column class intrested in.

print(class_df.head())

graph =  class_df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]

#graph.hist() #shows the histogram plot

#plt.show() #displays the plot

plt.scatter(class_df.ENGINESIZE,class_df.CO2EMISSIONS,color="red")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

#creating the train and test dataset

msk = np.random.rand(len(class_df)) > 0.7
print(msk)
train = class_df[msk]
test = class_df[~msk]

#plotting training data

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

#ploting test data

plt.scatter(test.ENGINESIZE,test.CO2EMISSIONS)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

regre_model = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regre_model.fit(train_x,train_y)

print ('Coefficients: ', regre_model.coef_)
print ('Intercept: ',regre_model.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regre_model.coef_[0][0]*train_x + regre_model.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# predicting 

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regre_model.predict(test_x)
print(test_y_hat)

#Evalution Metrics
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )