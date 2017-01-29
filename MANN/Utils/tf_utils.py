import tensorflow as tf
import numpy as np

def shared_float32(x, name=''):
	return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name)


def update_tensor(V, dim2, val):  # Update tensor V, with index(:,dim2[:]) by val[:]
	val = tf.cast(val, V.dtype)
	print V.get_shape().as_list(), dim2.get_shape().as_list(), val.get_shape().as_list()
	with tf.variable_scope('body', reuse=None):
		temp = tf.get_variable(name="temp", shape=V.get_shape().as_list()[1:],
							   initializer=tf.constant_initializer(0, dtype=V.dtype))

	def body((v, d2, chg), _):
		d2_int = tf.cast(d2, tf.int32)
		with tf.variable_scope('body', reuse=True):
			temp = tf.get_variable(name="temp", shape=v.get_shape().as_list(),
								   initializer=tf.constant_initializer(0, dtype=V.dtype))
		temp[:].assign(v)
		temp[d2_int].assign(chg)
		return temp, d2, chg

	Z, _, _ = tf.scan(body, elems=(V, dim2, val), name="Scan_Update")
	with tf.control_dependencies([Z]):
		return Z
