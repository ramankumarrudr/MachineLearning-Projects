import tensorflow as tf
import keras
import numpy as np
#A neural network is a set of functions that can learn patters

#building a neural network with one neuron

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])	#single neuron , input_shape is just one neuron

model.compile(optimizer='sgd',loss='mean_squared_error') # sgd -sarcastic gradient decent

x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0])
y = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0])

model.fit(x,y ,epochs=1)

print(model.predict([10.0]))