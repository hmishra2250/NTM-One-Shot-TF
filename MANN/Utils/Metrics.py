import tensorflow as tf
import numpy as np


# prediction is the argmax

def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], nb_classes=5, nb_samples_per_class=10, batch_size=1):
    accuracy_0 = tf.Variable(tf.zeros((batch_size, nb_samples_per_class), dtype=tf.float32))
    indices_0 = tf.Variable(tf.zeros((batch_size, nb_classes)), dtype=tf.float32)
    batch_range = tf.range(0, batch_size)

    def update_tensor(V, dim1, dim2, val):  # Update tensor V, with index(:,dim2[:]) by val[:]
        ix = tf.Variable(0)
        Z = None
        cond = lambda V, Z, d1, d2, ix: ix < d1

        def body(V, Z, d1, d2, ix):
            temp = tf.Variable(V[ix], validate_shape=False)
            temp = tf.scatter_update(temp, d2[ix], val[ix])
            if Z is not None:
                Z = tf.concat_v2([Z, temp], axis=0)
            else:
                Z = temp
            with tf.control_dependencies([Z]):
                return V, Z, d1, d2, ix + 1

        tf.while_loop(cond, body, [V, Z, dim1, dim2, ix])
        return Z

    def step_(p, t, acc, idx):
        acc = update_tensor(acc, batch_size, idx[batch_range, t], tf.equal(p, t))
        idx = update_tensor(idx, batch_size, t, 1)
        return (acc, idx)

    raw_accuracy, _ = tf.scan(step_, elems=[tf.transpose(predictions, perm=[1, 0]), tf.transpose(targets, perm=[1, 0])],
                              initializer=[accuracy_0, indices_0])
    accuracy = tf.reduce_mean(raw_accuracy / nb_classes, axis=0)
    return accuracy
