import tensorflow as tf
import numpy as np


def shared_float32(x, name=''):
    return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name)


def update_tensor(V, dim2, val):  # Update tensor V, with index(:,dim2[:]) by val[:]
    print 'Shapes Recieved in Update: V, dim, val are ==> ',V.get_shape().as_list(), dim2.get_shape().as_list(), val.get_shape().as_list()
    val = tf.cast(val, V.dtype)

    def body(_, (v, d2, chg)):
        print 'Shapes Recieved in Body of Update: v, d2, chg are ==> ', v.get_shape().as_list(), d2.get_shape().as_list(), chg.get_shape().as_list()
        d2_int = tf.cast(d2, tf.int32)
        if len(chg.get_shape().as_list()) == 0:
            chg = [chg]
        else:
            chg = tf.reshape(chg, shape=[1]+chg.get_shape().as_list())
        oob = lambda : tf.slice(tf.concat_v2([v[:d2_int], chg], axis=0), tf.range(0,len(v.get_shape().as_list())), v.get_shape().as_list())
        inb = lambda : tf.slice(tf.concat_v2([v[:d2_int], chg, v[d2_int + 1:]], axis=0), tf.constant(0,shape=[len(v.get_shape().as_list())]), v.get_shape().as_list())
        return tf.cond(tf.less(d2_int + 1, v.get_shape().as_list()[0]), inb, oob)

    Z = tf.scan(body, elems=(V, dim2, val), initializer=tf.constant(1, shape=V.get_shape().as_list()[1:], dtype=tf.float32), name="Scan_Update")
    return Z
