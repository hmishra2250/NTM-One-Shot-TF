import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #No logging TF

import tensorflow as tf
import numpy as np
import time

from MANN.Model import memory_augmented_neural_network
from MANN.Utils.Generator import OmniglotGenerator
from MANN.Utils.Metrics import accuracy_instance
from MANN.Utils.tf_utils import update_tensor

def omniglot():

    sess = tf.InteractiveSession()

    input_ph = tf.placeholder(dtype=tf.float32, shape=(16,50,400))   #(batch_size, time, input_dim)
    target_ph = tf.placeholder(dtype=tf.int32, shape=(16,50))     #(batch_size, time)(label_indices)

    ##Global variables for Omniglot Problem
    nb_reads = 4
    controller_size = 200
    memory_shape = (128,40)
    nb_class = 5
    input_size = 20*20
    batch_size = 16
    nb_samples_per_class = 10

    #Load Data
    generator = OmniglotGenerator(data_folder='./data/omniglot', batch_size=batch_size, nb_samples=nb_class, nb_samples_per_class=nb_samples_per_class, max_rotation=0., max_shift=0., max_iter=None)
    output_var, output_var_flatten, params = memory_augmented_neural_network(input_ph, target_ph, batch_size=batch_size, nb_class=nb_class, memory_shape=memory_shape, controller_size=controller_size, input_size=input_size, nb_reads=nb_reads)

    print 'Compiling the Model'
    

    with tf.variable_scope("Weights", reuse=True):
        W_key = tf.get_variable('W_key', shape=(nb_reads, controller_size, memory_shape[1]))
        b_key = tf.get_variable('b_key', shape=(nb_reads, memory_shape[1]))
        W_add = tf.get_variable('W_add', shape=(nb_reads, controller_size, memory_shape[1]))
        b_add = tf.get_variable('b_add', shape=(nb_reads, memory_shape[1]))
        W_sigma = tf.get_variable('W_sigma', shape=(nb_reads, controller_size, 1))
        b_sigma = tf.get_variable('b_sigma', shape=(nb_reads, 1))
        W_xh = tf.get_variable('W_xh', shape=(input_size + nb_class, 4 * controller_size))
        b_h = tf.get_variable('b_xh', shape=(4 * controller_size))
        W_o = tf.get_variable('W_o', shape=(controller_size + nb_reads * memory_shape[1], nb_class))
        b_o = tf.get_variable('b_o', shape=(nb_class))
        W_rh = tf.get_variable('W_rh', shape=(nb_reads * memory_shape[1], 4 * controller_size))
        W_hh = tf.get_variable('W_hh', shape=(controller_size, 4 * controller_size))
        gamma = tf.get_variable('gamma', shape=[1], initializer=tf.constant_initializer(0.95))

    params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]
    
    #output_var = tf.cast(output_var, tf.int32)
    target_ph_oh = tf.one_hot(target_ph, depth=generator.nb_samples)
    print 'Output, Target shapes: ',output_var.get_shape().as_list(), target_ph_oh.get_shape().as_list()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_var, target_ph_oh), name="cost")
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = opt.minimize(cost, var_list=params)

    #train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
    accuracies = accuracy_instance(tf.argmax(output_var, axis=2), target_ph, batch_size=generator.batch_size)
    sum_out = tf.reduce_sum(tf.reshape(tf.one_hot(tf.argmax(output_var, axis=2), depth=generator.nb_samples), (-1, generator.nb_samples)), axis=0)

    print 'Done'

    tf.summary.scalar('cost', cost)
    for i in range(generator.nb_samples_per_class):
    	tf.summary.scalar('accuracy-'+str(i), accuracies[i])
    
    merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('/tmp/tensorflow', graph=tf.get_default_graph())
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/', sess.graph)

    t0 = time.time()
    all_scores, scores, accs = [],[],np.zeros(generator.nb_samples_per_class)


    sess.run(tf.global_variables_initializer())

    print 'Training the model'



    try:
        for i, (batch_input, batch_output) in generator:
            feed_dict = {
                input_ph: batch_input,
                target_ph: batch_output
            }
            #print batch_input.shape, batch_output.shape
            train_step.run(feed_dict)
            score = cost.eval(feed_dict)
            acc = accuracies.eval(feed_dict)
            temp = sum_out.eval(feed_dict)
            summary = merged.eval(feed_dict)
            train_writer.add_summary(summary, i)
            print i, ' ',temp
            all_scores.append(score)
            scores.append(score)
            accs += acc
            if i>0 and not (i%100):
                print(accs / 100.0)
                print('Episode %05d: %.6f' % (i, np.mean(score)))
                scores, accs = [], np.zeros(generator.nb_samples_per_class)


    except KeyboardInterrupt:
        print time.time() - t0
        pass

if __name__ == '__main__':
    omniglot()


