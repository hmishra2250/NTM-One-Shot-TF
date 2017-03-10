import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


from MANN.Utils.Metrics import accuracy_instance
import tensorflow as tf
import numpy as np
import copy

x = [0,0,0,0,0]*10
y = [0,1,2,3,4]*10
np.random.shuffle(y)
x = np.append([x],[x],axis=0)
y = np.append([y], [y], axis=0)

p = tf.constant(x)
t = tf.constant(y)

sess = tf.InteractiveSession()

zz = accuracy_instance(p, t, batch_size=2)

sess.run(zz)

print p[0].eval()
print t[0].eval()

print zz.eval()

print tf.equal(p,t).eval()