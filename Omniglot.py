import tensorflow as tf
import numpy as np
import time

from MANN.Model import memory_augmented_neural_network
from MANN.Utils.Generator import OmniglotGenerator
from MANN.Utils.Metrics import accuracy_instance

def omniglot():

    sess = tf.InteractiveSession()

    input_ph = tf.placeholder(dtype=tf.float32, shape=(16,50,400))   #(batch_size, time, input_dim)
    target_ph = tf.placeholder(dtype=tf.int32, shape=(16,50))     #(batch_size, time)(label_indices)

    #Load Data
    generator = OmniglotGenerator(data_folder='./data/omniglot', batch_size=16, nb_samples=5, nb_samples_per_class=10, max_rotation=0., max_shift=0., max_iter=1000)
    output_var, output_var_flatten, params = memory_augmented_neural_network(input_ph, target_ph, batch_size=generator.batch_size, nb_class=generator.nb_samples, memory_shape=(128,40), controller_size=200, input_size=20*20, nb_reads=4)

    print 'Compiling the Model'

    target_ph = tf.one_hot(tf.reshape(target_ph, shape=[-1, 1]), depth=generator.nb_samples)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_var_flatten, target_ph))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
    accuracies = accuracy_instance(tf.argmax(output_var, axis=2), target_ph, batch_size=generator.batch_size)

    print 'Training the model'

    t0 = time.time()
    all_scores, scores, accs = [],[],np.zeros(generator.nb_samples_per_class)


    sess.run(tf.global_variables_initializer())

    try:
        for i, (batch_input, batch_output) in generator:
            feed_dict = {
                input_ph: batch_input,
                target_ph: batch_output
            }
            train_step.run(feed_dict)
            score = cost.eval(feed_dict)
            acc = accuracies.eval(feed_dict)
            all_scores.append(score)
            scores.append(score)
            accuracies += acc

            if i>0 and not (i%20):
                print('Episode %05d: %.6f' % (i, np.mean(score)))
                print(accs / 100.)
                scores, accs = [], np.zeros(generator.nb_samples_per_class)


    except KeyboardInterrupt:
        print time.time() - t0
        pass

if __name__ == '__main__':
    omniglot()


