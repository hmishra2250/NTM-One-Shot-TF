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
        # type: (object, object, object) -> object
        return [x[:,n*size:(n+1)*size] for n in range(nb_slice)]

    def step(X_t, M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, wu_tm1, ix):

        x_t = tf.transpose(X_t, perm=[1, 0, 2])[ix]
        preactivations = tf.matmul(x_t,W_xh) + tf.matmul(r_tm1,W_rh) + tf.matmul(h_tm1,W_hh) + b_h
        gf_, gi_, go_, u_ = slice_equally(preactivations, controller_size, 4)
        gf = tf.sigmoid(gf_)
        gi = tf.sigmoid(gi_)
        go = tf.sigmoid(go_)
        u = tf.sigmoid(u_)

        c_t = gf*c_tm1 + gi*u
        h_t = go * tf.tanh(c_t)  #(batch_size, controller_size)

        k_t = tf.tanh(tf.matmul(h_t,W_key) + b_key)  #(batch_size, nb_reads, memory_shape[1])
        a_t = tf.tanh(tf.matmul(h_t,W_add) + b_add)
        sigma_t = tf.sigmoid(tf.matmul(h_t,W_sigma) + b_sigma)  #(batch_size, nb_reads, 1)

        _,temp_indices = tf.nn.top_k(wu_tm1, memory_shape[0])
        wlu_tm1 = tf.slice(temp_indices, [0,0], [batch_size,nb_reads])    #(batch_size, nb_reads)

        ww_t = tf.reshape(tf.matmul(sigma_t, wr_tm1), (batch_size*nb_reads, memory_shape[0]))
        ww_t = tf.scatter_add(ww_t, [tf.range(0, nb_reads*memory_shape[0]), tf.reshape(wlu_tm1,[-1])],1.0 - sigma_t)
        ww_t = tf.reshape((batch_size, nb_reads, memory_shape[0]))

        M_t = tf.scatter_update(M_tm1, [tf.range(0,batch_size),wlu_tm1[:,0],], 0.)
        M_t = tf.add(M_t, tf.batch_matmul(tf.transpose(ww_t, perm=[0,2,1]   ), a_t))   #(batch_size, memory_size[0], memory_size[1])
        K_t = cosine_similarity(k_t, M_t)

        wr_t = tf.nn.softmax(tf.reshape(K_t, (batch_size*nb_reads, memory_shape[0])))
        wr_t = tf.reshape(wr_t, (batch_size, nb_reads, memory_shape[0]))    #(batch_size, nb_reads, memory_size[0])

        wu_t = gamma * wu_tm1 + tf.sum(wr_t, axis=1) + tf.sum(ww_t, axis=1) #(batch_size, memory_size[0])

        r_t = tf.reshape(tf.batch_matmul(wr_t, M_t),[batch_size,-1])
        ix = tf.add(ix,tf.constant(1,dtype=tf.int32))  #incrementing index

        return (M_t, c_t, h_t, r_t, wr_t, wu_t, ix)

    #Model Part:
    sequence_length_var = target_var.shape[1]   #length of the input
    output_shape_var = (batch_size * sequence_length_var, nb_class)     #(batch_size*sequence_length_vat,nb_class)

    # Input concat with time offset
    one_hot_target_flattened = tf.one_hot(output_shape_var, depth=nb_class)
    one_hot_target = tf.reshape(one_hot_target_flattened, (batch_size, sequence_length_var, nb_class))    #(batch_size, sequence_var_length, nb_class)
    offset_target_var = tf.concat_v2([tf.zeros_like(tf.expand_dims(one_hot_target[:,0],1)),one_hot_target[:,:-1]],axis=1)   #(batch_size, sequence_var_length, nb_class)
    l_input_var = tf.concat_v2([input_var,offset_target_var],axis=2)    #(batch_size, sequence_var_length, input_size+nb_class)

    ix = tf.variable(0,dtype=tf.int32)
    cond = lambda M_0, c_0, h_0, r_0, wr_0, wu_0, ix: ix < sequence_length_var
    l_ntm_var = tf.while_loop(cond, body=step,loop_vars=[M_0, c_0, h_0, r_0, wr_0, wu_0, ix])   #Set of all above parameters, as list
    l_ntm_output_var = tf.transpose(tf.concatenate(l_ntm_var[2:4], axis=2), perm=[1, 0, 2])     #h_t & r_t

    output_var_preactivation = tf.add(tf.matmul(l_ntm_output_var,W_o), b_o)
    output_var_flatten = tf.nn.softmax(tf.reshape(output_var_preactivation, output_shape_var))
    output_var = tf.reshape(output_var_flatten, output_var_preactivation.shape)

    #Parameters
    params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]

    return output_var, output_var_flatten, params