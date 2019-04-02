import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch, logs={}):
		if(logs.get('loss')<0.15):
			print("\nloss is low and cancelling training")
			self.model.stop_training=True

callbacks = myCallback()		

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()	#training labels are refered in numbers

plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])


# data normalizing

train_images = train_images / 255.0
test_images	= test_images / 255.0

#Neural Network with one hidden layer with 128 neurons
model = keras.Sequential([

		keras.layers.Flatten(),
		keras.layers.Dense(128, activation = tf.nn.relu),
		keras.layers.Dense(64, activation = tf.nn.relu),
		keras.layers.Dense(32, activation = tf.nn.relu),
		keras.layers.Dense(16, activation = tf.nn.relu),
		keras.layers.Dense(10, activation = tf.nn.softmax)	# output should be always the size of class

	])


model.compile(optimizer = tf.train.AdamOptimizer(), loss= 'sparse_categorical_crossentropy')

model.fit(train_images,train_labels,epochs=5,callbacks=[callbacks])

model.evaluate(test_images,test_labels)

