'''
Created on Oct 19 16:45 2018
@author :lilu
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:56:26 2018

@author: 10724
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

tf.app.flags.DEFINE_float('alpha',0.01,'weight to adjust the ace loss')
tf.app.flags.DEFINE_integer('batchsize',1024,'batch size')
FLAGS = tf.app.flags.FLAGS

alpha = FLAGS.alpha
batchsize = FLAGS.batchsize

#input_x = tf.placeholder(tf.float32,shape=None)
#input_y = tf.placeholder(tf.float32,shape=None)

input_x = np.random.normal(size=[32,120])
input_y = np.random.normal(size=[32,101])
input_y[input_y>0] = 1
input_y[input_y<=0] = 0
input_x = tf.convert_to_tensor(input_x,dtype=tf.float32)
input_y = tf.convert_to_tensor(input_y,dtype=tf.float32)


d = 120 # the feature dimention of input_x
l  =101 # the num of labels

#initial parameters
layers = 4
dims = [50,50,50,50]
with tf.variable_scope('input_x'):
    w = [None]*layers
    b = [None]*layers
    for i in range(layers):
        if i==0:
            weights = tf.random_uniform([d,dims[0]],minval = -np.sqrt(6)/(np.sqrt(d+dims[0])), \
                                        maxval=np.sqrt(6)/(np.sqrt(d+dims[0])))
        else:
            weights = tf.random_uniform([dims[i-1],dims[i]],minval = -np.sqrt(6)/(np.sqrt(dims[i-1]+dims[i])), \
                                        maxval=np.sqrt(6)/(np.sqrt(dims[i-1]+dims[i])))
        w[i] = tf.Variable(weights)
        b[i] = tf.Variable(tf.constant(0.001,shape=[dims[i]]))

with tf.variable_scope('input_y'):
    ytabel = tf.Variable(tf.random_uniform([l,dims[-1]],minval = -np.sqrt(6)/(np.sqrt(l+dims[-1])), \
                                        maxval=np.sqrt(6)/(np.sqrt(l+dims[-1]))))
    biase = tf.get_variable('y_biase',initializer=0.001*tf.ones(l))

feax = [input_x]    
for i in range(layers):
    tmp = tf.matmul(feax[i],w[i])+b[i]
    tmp = utils.batch_norm(tmp, phase)
    tmp = tf.nn.relu(tmp)
    feax.append(tmp)
logits = tf.matmul(feax[-1],ytabel,transpose_b=True)+biase
logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y,logits=logits))

ytable_inner = tf.matmul(ytabel,ytabel,transpose_b=True)
y_countx = tf.multiply(tf.matmul(input_y,input_y,transpose_a=True),1-tf.diag(tf.ones(l)))
corrloss = -1*tf.reduce_sum(tf.multiply(ytable_inner,y_countx))/tf.reduce_sum(y_countx)
y_count = tf.multiply(tf.matmul(input_y,input_y,transpose_a=True),tf.diag(tf.ones(l)))
y_cov = tf.matmul(tf.matmul(tf.transpose(ytabel),y_count),ytabel)/tf.cast(tf.reduce_sum(y_count) - 1, tf.float32)
traceloss = 0.5 * tf.reduce_sum(tf.reshape(y_cov, [-1]) * tf.reshape(y_cov, [-1]))
totalloss = (1-alpha)*logloss+alpha*(corrloss+traceloss)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)
loglossv,corrlossv,tracelossv = sess.run([logloss,corrloss,traceloss])















    
    
