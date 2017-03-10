import tensorflow as tf
import numpy as np
import time


def omniglot():

    sess = tf.InteractiveSession()

    """    def wrapper(v):
        return tf.Print(v, [v], message="Printing v")

    v = tf.Variable(initial_value=np.arange(0, 36).reshape((6, 6)), dtype=tf.float32, name='Matrix')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    temp = tf.Variable(initial_value=np.arange(0, 36).reshape((6, 6)), dtype=tf.float32, name='temp')
    temp = wrapper(v)
    #with tf.control_dependencies([temp]):
    temp.eval()
    print 'Hello'"""

    def update_tensor(V, dim2, val):  # Update tensor V, with index(:,dim2[:]) by val[:]
        val = tf.cast(val, V.dtype)
        def body(_, (v, d2, chg)):
            d2_int = tf.cast(d2, tf.int32)
            return tf.slice(tf.concat_v2([v[:d2_int],[chg] ,v[d2_int+1:]], axis=0), [0], [v.get_shape().as_list()[0]])
        Z = tf.scan(body, elems=(V, dim2, val), initializer=tf.constant(1, shape=V.get_shape().as_list()[1:], dtype=tf.float32), name="Scan_Update")
        return Z


    print 'Compiling the Model'

    tt1 = tf.Variable(initial_value=np.arange(0, 36).reshape((6, 6)), dtype=tf.float32, name='Matrix')
    ix = tf.Variable(initial_value=np.arange(0, 6), name='Indices')
    val = tf.Variable(initial_value=np.arange(100, 106), name='Values', dtype=tf.float32)

    tt = tf.concat_v2([tt1[:3], tf.reshape(tf.range(0,6,dtype=tf.float32),shape=(1,6)), tt1[3:]], axis=0)
    print tt1[:3].get_shape().as_list()

    """op = tt1[4].assign(val)
    sess.run(tf.global_variables_initializer())
    sess.run(op)
    print tt1.eval()"""

    op = tt1.assign(update_tensor(tt1, ix, val))
    val = tf.Print(val, [val], "This works fine")


    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    print 'Training the model'

    print tt.eval()
    writer = tf.summary.FileWriter('/tmp/tensorflow', graph=tf.get_default_graph())
    #tf.scalar_summary('cost', cost)

    print 'tt1: ',tt1.eval()
    print 'ix: ',ix.eval()
    print 'val: ',val.eval()

    sess.run(op)
    print 'After run\n', tt1.eval()
    #with tf.control_dependencies([op]):
    #    print '********************','\n',tt1.eval(),'\n', op.eval()



if __name__ == '__main__':
    omniglot()


