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

    #Load Data
    generator = OmniglotGenerator(data_folder='./data/omniglot', batch_size=16, nb_samples=5, nb_samples_per_class=10, max_rotation=0., max_shift=0., max_iter=1000)
    output_var, output_var_flatten, params = memory_augmented_neural_network(input_ph, target_ph, batch_size=generator.batch_size, nb_class=generator.nb_samples, memory_shape=(128,40), controller_size=200, input_size=20*20, nb_reads=4)

    print 'Compiling the Model'

    #output_var = tf.cast(output_var, tf.int32)
    target_ph_flatten = tf.one_hot(tf.reshape(target_ph, shape=[-1, 1]), depth=generator.nb_samples)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_var_flatten, target_ph_flatten), name="cost")
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
    accuracies = accuracy_instance(tf.argmax(output_var, axis=2), target_ph, batch_size=generator.batch_size)

    print 'Done'

    sess.run(tf.global_variables_initializer())

    print 'Training the model'

    writer = tf.summary.FileWriter('/tmp/tensorflow', graph=tf.get_default_graph())
    tf.summary.scalar('cost', cost)


    t0 = time.time()
    all_scores, scores, accs = [],[],np.zeros(generator.nb_samples_per_class)



    try:
        for i, (batch_input, batch_output) in generator:
            feed_dict = {
                input_ph: batch_input,
                target_ph: batch_output
            }
            print batch_input.shape, batch_output.shape
            train_step.run(feed_dict)
            score = cost.eval(feed_dict)
            acc = accuracies.eval(feed_dict)
            all_scores.append(score)
            scores.append(score)
            accs += acc

            if i>0 and not (i%2):
                print(accs / 100.0)
                print('Episode %05d: %.6f' % (i, np.mean(score)))
                scores, accs = [], np.zeros(generator.nb_samples_per_class)


    except KeyboardInterrupt:
        print time.time() - t0
        pass

if __name__ == '__main__':
    omniglot()


