import tensorflow as tf

def cosine_similarity(x, y, eps=1e-6):
	z = tf.batch_matmul(x, tf.transpose(y, perm=[0,2,1]))
	z /= tf.sqrt(tf.multiply(tf.expand_dims(tf.reduce_sum(tf.multiply(x,x), 2), 2),tf.expand_dims(tf.reduce_sum(tf.multiply(y,y), 2), 1)) + eps)
	
	return z
