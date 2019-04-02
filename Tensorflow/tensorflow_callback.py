import tensorflow as tf
import keras
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch, logs={})
	if(logs.get('loss')<0.25):
		print("\nloss is low and cancelling training")
		self.model.stop_training=True