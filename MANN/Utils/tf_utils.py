import tensorflow as tf
import numpy as np

def shared_float32(x, name=''):
	return tf.Variable(np.asarray(x, dtype=tf.float32), name=name)
