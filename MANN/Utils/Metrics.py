import tensorflow as tf
import numpy as np

#prediction is the argmax

def accuracy_instance(predictions, targets, n=[1,2,3,4,5,10], nb_classes=5, nb_samples_per_class=10, batch_size=1):
    accuracy_0 = tf.Variable(tf.zeros((batch_size, nb_samples_per_class),dtype=tf.float32))
    indices_0 = tf.Variable(tf.zeros((batch_size, nb_classes)), dtype=tf.float32)
    batch_range = tf.range(0,batch_size)
    def step_(p, t, acc, idx):
        ix = tf.Variable(0, dtype=tf.int32)
        #def body
        acc = tf.scatter_add(acc, [])

