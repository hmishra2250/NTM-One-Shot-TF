import tensorflow
import tensorflow as tf
import numpy as np
import sys

def shared_glorot_uniform(shape, dtype=tf.float32, name='', n=None):
	if isinstance(shape,int):
		high = np.sqrt(6. / shape)
		shape = [shape]
	else:
		high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
	shape = shape if n is None else [	n] + list(shape)
	return tf.Variable(tf.random_uniform(shape, minval=-high, maxval=high, dtype=dtype, name=name))
	
def shared_zeros(shape, dtype=tf.float32, name='', n=None):
	shape = shape if n is None else (n,) + tuple(shape)
	return tf.Variable(tf.zeros(shape, dtype=dtype), name=name)
	
def shared_one_hot(shape, dtype=tf.float32, name='', n=None):
	shape = (shape,) if isinstance(shape,int) else shape
	shape = shape if n is None else (n,) + shape
	initial_vector = np.zeros(shape, dtype=np.float32)
	initial_vector[...,0] = 1
	return tf.Variable(tf.cast(initial_vector, tf.float32), name=name)
	
def weight_and_bias_init(shape, dtype=tf.float32, name='', n=None):
	return (shared_glorot_uniform(shape, dtype=dtype, name='W_' + name, n=n), \
		shared_zeros((shape[1],), dtype=dtype, name='b_' + name, n=n))
