import tensorflow as tf
import numpy as np

from .Utils.init import weight_and_bias_init, shared_glorot_uniform, shared_one_hot
from .Utils.similarities import cosine_similarity
from .Utils.tf_utils import shared_float32


def memory_augmented_neural_network(input_var, target_var, \
                                    batch_size=16, nb_class=5, memory_shape=(128, 40), \
                                    controller_size=200, input_size=20 * 20, nb_reads=4):
    ## input_var has dimensions (batch_size, time, 	input_dim)
    ## target_var has dimensions (batch_size, time) (label indices)

    M_0 = shared_float32(1e-6 * np.ones((batch_size,) + memory_shape), name='memory')
    c_0 = shared_float32(np.zeros((batch_size, controller_size)), name='memory_cell_state')
    h_0 = shared_float32(np.zeros((batch_size, controller_size)), name='hidden_state')
    r_0 = shared_float32(np.zeros((batch_size, nb_reads * memory_shape[1])), name='read_vector')
    wr_0 = shared_one_hot((batch_size, nb_reads, memory_shape[0]), name='wr')
    wu_0 = shared_one_hot((batch_size, memory_shape[0]), name='wu')

    W_key, b_key = weight_and_bias_init((controller_size, memory_shape[1]), name='key', n=nb_reads)
    W_add, b_add = weight_and_bias_init((controller_size, memory_shape[1]), name='add', n=nb_reads)
    W_sigma, b_sigma = weight_and_bias_init((controller_size, 1), name='sigma', n=nb_reads)

    W_xh, b_h = weight_and_bias_init((input_size + nb_class, 4 * controller_size),name='xh')
    W_rh = shared_glorot_uniform((nb_reads * memory_shape[1], 4 * controller_size), name='W_rh')
    W_hh = shared_glorot_uniform((controller_size, 4*controller_size),name='W_hh')
    W_o, b_o = weight_and_bias_init((controller_size + nb_reads * memory_shape[1], nb_class),name='o')
    gamma = 0.95

    def slice_equally(x, size, nb_slice):
        return [x[:,n*size:(n+1)*size] for n in range(nb_slice)]

    def step(x_t, M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, wu_tm1):

        preactivations = tf.mul(x_t,W_xh) + tf.mul(r_tm1,W_rh) + tf.mul(h_tm1,W_hh) + b_h
        gf_, gi_, go_, u_ = slice_equally(preactivations, controller_size, 4)
        gf = tf.sigmoid(gf_)
        gi = tf.sigmoid(gi_)
        go = tf.sigmoid(go_)
        u = tf.sigmoid(u_)

        c_t = gf*c_tm1 + gi*u
        h_t = go * tf.tanh(c_t)  #(batch_size, controller_size)

        k_t = tf.tanh(tf.mul(h_t,W_key) + b_key)  #(batch_size, nb_reads, memory_shape[1])
        a_t = tf.tanh(tf.mul(h_t,W_add) + b_add)
        sigma_t = tf.sigmoid(tf.mul(h_t,W_sigma) + b_sigma)  #(batch_size, nb_reads, 1)

        _,temp_indices = tf.nn.top_k(wu_tm1, memory_shape[0])
        wlu_tm1 = tf.slice(temp_indices, [0,0], [batch_size,nb_reads])    #(batch_size, nb_reads)

        ww_t = tf.reshape(tf.mul(sigma_t, wr_tm1), (batch_size*nb_reads, memory_shape[0]))
        ww_t = tf.scatter_add(ww_t, [tf.range(0, nb_reads*memory_shape[0]), tf.reshape(wlu_tm1,[-1])],1.0 - sigma_t)
        ww_t = tf.reshape((batch_size, nb_reads, memory_shape[0]))

        M_t = tf.scatter_update(M_tm1, [tf.range(0,batch_size),wlu_tm1[:,0],], 0.)
        M_t = tf.add(M_t, tf.batch_matmul(tf.transpose(ww_t, perm=[0,2,1]), a_t))   #(batch_size, memory_size[0], memory_size[1])
        K_t = cosine_similarity(k_t, M_t)

        wr_t = tf.nn.softmax(tf.reshape(K_t, (batch_size*nb_reads, memory_shape[0])))
        wr_t = tf.reshape(wr_t, (batch_size, nb_reads, memory_shape[0]))    #(batch_size, nb_reads, memory_size[0])

        wu_t = gamma * wu_tm1 + tf.sum(wr_t, axis=1) + tf.sum(ww_t, axis=1) #(batch_size, memory_size[0])

        r_t = tf.reshape(tf.batch_matmul(wr_t, M_t),[batch_size,-1])

        return (M_t, c_t, h_t, r_t, wr_t, wu_t)

    #Model Part:
    sequence_length_var = target_var.shape[1]
    output_shape_var = (batch_size * sequence_length_var, nb_class)

