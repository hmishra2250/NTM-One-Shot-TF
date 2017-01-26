import tensorflow as tf
import numpy as np


# prediction is the argmax

def accuracy_instance(predictions, targets, n=[1, 2, 3, 4, 5, 10], nb_classes=5, nb_samples_per_class=10, batch_size=1):
    targets = tf.cast(targets, predictions.dtype)
    accuracy_0 = tf.Variable(tf.zeros((batch_size, nb_samples_per_class), dtype=tf.float32))
    indices_0 = tf.Variable(tf.zeros((batch_size, nb_classes+1)), dtype=tf.float32)
    batch_range = tf.range(0, batch_size)
    #print '================================>>>>>>>>>>', predictions.dtype, targets.dtype

    def update_tensor(V, dim1, dim2, val):  # Update tensor V, with index(:,dim2[:]) by val[:]
        val = tf.cast(val, V.dtype)
        print val.dtype
        ix = tf.Variable(0, dtype=tf.int32)
        Z = tf.Variable(tf.zeros_like(V), dtype=tf.float32, name="Z_all")
        cond = lambda V, d1, d2, ix: ix < d1

        def body(V, d1, d2, ix):
            ix_int = tf.cast(ix, tf.int32)
            d2_int = tf.cast(d2, tf.int32)
            Z = tf.get_variable("Z_all")
            temp = tf.Variable(V[ix_int], validate_shape=False)
            temp = tf.scatter_update(temp, d2_int[ix_int], val[ix_int])
            Z[ix].assign(temp)
            tf.get_variable_scope().reuse_variables()
            return V, d1, d2, ix + 1

        temp = tf.while_loop(cond, body, [V, dim1, dim2, ix], name="While_Metric_Update")
        with tf.control_dependencies([Z]):
            return Z

    def step_((acc, idx), (p, t)):
        p = tf.cast(p, tf.int32)
        t = tf.cast(t, tf.int32)
        print 'brrrrrrrrrrr=======================**********>>>>', t.get_shape().as_list(), p.get_shape().as_list(), batch_range.get_shape().as_list()
        acc = update_tensor(acc, batch_size, tf.gather_nd(idx,tf.pack([batch_range, t], axis=1)), tf.equal(p, t))
        #print '=======================**********>>>>'
        tt = batch_range.get_shape().as_list()[0]
        idx = update_tensor(idx, batch_size, t, tf.constant(1, shape=[batch_range.get_shape().as_list()[0]]))
        print 'Done _Scan'
        with tf.control_dependencies([acc,idx]):
            return [acc, idx]

    #print '=======================**********>>>>', targets.get_shape().as_list(), predictions.get_shape().as_list(), batch_range.get_shape().as_list()
    print '************************============================', accuracy_0.dtype, indices_0.dtype
    print '************************============================', predictions.dtype, targets.dtype
    tt1 = tf.transpose(predictions, perm=[1, 0])
    tt2 = tf.transpose(targets, perm=[1, 0])
    print '*+*', tt1.dtype, tt2.dtype
    raw_accuracy, _ = tf.scan(step_, elems=(tf.transpose(predictions, perm=[1, 0]), tf.transpose(targets, perm=[1, 0])),initializer=[accuracy_0, indices_0], name="Scan_Metric_Last")
    accuracy = tf.reduce_mean(raw_accuracy / nb_classes, axis=0)
    with tf.control_dependencies([accuracy]):
        print 'Done accuracy'
        return accuracy
