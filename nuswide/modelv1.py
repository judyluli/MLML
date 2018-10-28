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
import utils
import input

tf.reset_default_graph()

tf.app.flags.DEFINE_float('alpha',0.001,'weight to adjust the ace loss')
tf.app.flags.DEFINE_integer('batchsize',1024,'batch size')
tf.app.flags.DEFINE_integer('epoches',50,'poches')
FLAGS = tf.app.flags.FLAGS

alpha = FLAGS.alpha
batch_size = FLAGS.batchsize
epoches = FLAGS.epoches

input_x = tf.placeholder(tf.float32,shape=None)
input_y = tf.placeholder(tf.float32,shape=None)
phase = tf.placeholder(tf.bool, name='phase')

def drop(data,rate=0.4):
    result = data.copy()
    [m,n] = data.shape
    for i in range(m):
        for j in range(n):
            if result[i,j] ==1:
                result[i,j] = np.random.binomial(1,1-rate)
    return result

data_x = input.readx('/data/users/lulusee/nus/Low_Level_Features','Train')
data_y = input.ready('/data/users/lulusee/nus/Tags/Train_Tags1k.dat')
test_x = input.readx('/data/users/lulusee/nus/Low_Level_Features','Test')
test_y = input.ready('/data/users/lulusee/nus/Tags/Test_Tags1k.dat')
num_samples = len(data_x)
data_y = drop(data_y)

d = data_x.shape[1] # the feature dimention of input_x
l = data_y.shape[1] # the num of labels

y_countx = tf.convert_to_tensor(np.dot(data_y.T,data_y)*(1-np.diag(np.ones(l))),dtype=tf.float32)
y_count = tf.convert_to_tensor(np.dot(data_y.T,data_y)*(np.diag(np.ones(l))),dtype=tf.float32)


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

#build computation graph
feax = [input_x]
for i in range(layers):
    tmp = tf.matmul(feax[i],w[i])+b[i]
    tmp = utils.batch_norm(tmp, phase)
    tmp = tf.nn.relu(tmp)
    feax.append(tmp)
logits = tf.matmul(feax[-1],ytabel,transpose_b=True)+biase
logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y,logits=logits))

ytable_inner = tf.matmul(ytabel,ytabel,transpose_b=True)
corrloss = -1*tf.reduce_sum(tf.multiply(ytable_inner,y_countx))/tf.reduce_sum(y_countx)
y_cov = tf.matmul(tf.matmul(tf.transpose(ytabel),y_count),ytabel)/tf.cast(tf.reduce_sum(y_count) - 1, tf.float32)
traceloss = 0.5 * tf.reduce_sum(tf.reshape(y_cov, [-1]) * tf.reshape(y_cov, [-1]))
aceloss = corrloss+traceloss
totalloss = (1-alpha)*logloss+alpha*aceloss

optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(totalloss)


prediction = tf.nn.sigmoid(logits)
def evaluation(pre,truth):
    pre[range(len(pre)),np.argmax(pre,axis=1)] = 1
    pre[pre<1] = 0
    precision = np.sum(pre*truth)/np.sum(pre)
    recall = np.sum(pre*truth)/np.sum(truth)
    fscore = 2*precision*recall/(precision+recall)
    acc = np.sum(pre==truth)/(truth.shape[0]*truth.shape[1])
    return acc,precision,fscore

testfscore = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps_per_epoch = int(num_samples/batch_size)
    for i in range(epoches):
        idx = list(range(len(data_x)))
        np.random.shuffle(idx)
        data_x = data_x[idx]
        data_y = data_y[idx]
        for j in range(steps_per_epoch):
            rand_x = data_x[j*batch_size:min((j+1)*batch_size,num_samples)]
            rand_y = data_y[j*batch_size:min((j+1)*batch_size,num_samples)]
            sess.run(train_step,feed_dict={input_x:rand_x,input_y:rand_y,phase:True})
        loglossv,acelossv = sess.run([logloss,aceloss],feed_dict={input_x:data_x,input_y:data_y,phase:False})
        trainlossv = sess.run(totalloss,feed_dict={input_x:data_x,input_y:data_y,phase:False})
        testlossv = sess.run(totalloss,feed_dict={input_x:test_x,input_y:test_y,phase:False})
        predictionv = sess.run(prediction,feed_dict={input_x:test_x,input_y:test_y,phase:False})
        acc,precision,fscore = evaluation(predictionv,test_y)
        print('epoch{}---trainloss{},logloss{},aceloss{}'.format(i,trainlossv,loglossv,acelossv))
        print('epoch{}---testloss{},testacc{},testprecision{},testfscore{}'.format(i,testlossv,acc,precision,fscore))
        testfscore.append(fscore)
avgfscore = np.mean(np.array(testfscore[-10:]))
print('average fsoce:%f'%(avgfscore))
