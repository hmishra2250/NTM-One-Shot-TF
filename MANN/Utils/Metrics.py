import tensorflow as tf
import numpy as np
from .tf_utils import update_tensor


# prediction is the argmax

def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], nb_classes=5, nb_samples_per_class=10, batch_size=1):
    targets = tf.cast(targets, predictions.dtype)
    with tf.variable_scope("Metric_step_var"):
        accuracy = tf.get_variable(name="accuracy", shape=(batch_size, nb_samples_per_class), initializer=tf.constant_initializer(0), dtype=tf.float32)
        indices = tf.get_variable(name="indices", shape=(batch_size, nb_classes+1), initializer=tf.constant_initializer(0), dtype=tf.float32)

    def step_((acc, ix), (p, t)):
        with tf.variable_scope("Metric_step_var", reuse=True):
            accuracy = tf.get_variable(name="accuracy", shape=(batch_size, nb_samples_per_class),
                                       initializer=tf.constant_initializer(0), dtype=tf.float32)
            indices = tf.get_variable(name="indices", shape=(batch_size, nb_classes + 1),
                                      initializer=tf.constant_initializer(0), dtype=tf.float32)

        ix = tf.cast(ix, dtype=tf.int32)
        p = tf.cast(p, tf.int32)
        t = tf.cast(t, tf.int32)
        with tf.variable_scope("Acc_Upd"):
            accuracy.assign(update_tensor(accuracy, tf.gather_nd(indices,tf.pack([tf.range(0,p.get_shape().as_list()[0]), t], axis=1)), tf.equal(p, t)))
        with tf.variable_scope("Index_Upd"):
            indices.assign(update_tensor(indices, t, tf.constant(1, shape=[p.get_shape().as_list()[0]])))
        return [accuracy, ix+1]

    ix = tf.constant(0, dtype=tf.int32)
    _, _ = tf.scan(step_, elems=(tf.transpose(predictions, perm=[1, 0]), tf.transpose(targets, perm=[1, 0])),initializer=[accuracy, ix], name="Scan_Metric_Last")
    with tf.variable_scope("Metric_step_var", reuse=True):
        accuracy = tf.get_variable(name="accuracy", shape=(batch_size, nb_samples_per_class),
                                   initializer=tf.constant_initializer(0), dtype=tf.float32)
    accuracy = tf.reduce_mean(accuracy / nb_classes, axis=0)
    with tf.control_dependencies([accuracy]):
        print 'Done accuracy'
        return accuracy
