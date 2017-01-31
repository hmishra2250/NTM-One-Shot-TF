import tensorflow as tf
import numpy as np
from .tf_utils import update_tensor


# prediction is the argmax

def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], nb_classes=5, nb_samples_per_class=10, batch_size=1):
    targets = tf.cast(targets, predictions.dtype)

    accuracy = tf.constant(value=0, shape=(batch_size, nb_samples_per_class), dtype=tf.float32)
    indices = tf.constant(value=0, shape=(batch_size, nb_classes+1), dtype=tf.float32)

    def step_((accuracy, indices), (p, t)):
        """with tf.variable_scope("Metric_step_var", reuse=True):
            accuracy = tf.get_variable(name="accuracy", shape=(batch_size, nb_samples_per_class),
                                       initializer=tf.constant_initializer(0), dtype=tf.float32)
            indices = tf.get_variable(name="indices", shape=(batch_size, nb_classes + 1),
                                      initializer=tf.constant_initializer(0), dtype=tf.float32)"""

        p = tf.cast(p, tf.int32)
        t = tf.cast(t, tf.int32)
        ##Accuracy Update
        batch_range = tf.cast(tf.range(0, batch_size), dtype=tf.int32)
        gather = tf.cast(tf.gather_nd(indices,tf.pack([tf.range(0,p.get_shape().as_list()[0]), t], axis=1)), tf.int32)
        index = tf.cast(tf.pack([batch_range, gather], axis=1), dtype=tf.int64)
        val = tf.cast(tf.equal(p, t), tf.float32)
        delta = tf.SparseTensor(indices=index, values=val, shape=tf.cast(accuracy.get_shape().as_list(), tf.int64))
        accuracy = accuracy + tf.sparse_tensor_to_dense(delta)
        ##Index Update
        index = tf.cast(tf.pack([batch_range, t], axis=1), dtype=tf.int64)
        val = tf.constant(1.0, shape=[batch_size])
        delta = tf.SparseTensor(indices=index, values=val, shape=tf.cast(indices.get_shape().as_list(), dtype=tf.int64))
        indices = indices + tf.sparse_tensor_to_dense(delta)
        return [accuracy, indices]

    accuracy, indices = tf.scan(step_, elems=(tf.transpose(predictions, perm=[1, 0]), tf.transpose(targets, perm=[1, 0])),initializer=[accuracy, indices], name="Scan_Metric_Last")

    accuracy = accuracy[-1]

    accuracy = tf.reduce_mean(accuracy / nb_classes , axis=0)

    return accuracy
