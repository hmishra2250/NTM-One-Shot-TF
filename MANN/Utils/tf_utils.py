import tensorflow as tf
import numpy as np

def shared_float32(x, name=''):
	return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name)
