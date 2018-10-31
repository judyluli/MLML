import tensorflow as tf
from . import utils
import numpy as np
import time
from tensorflow.python.ops import array_ops
# this is test for github
class ACE():
    def __init__(self,layers=[50,50,50,50],optimizer=tf.train.AdamOptimizer(0.001),num_features=None,num_labels=None,seed=None):
        self.layers = layers
        self.optimizer = optimizer
        self.num_features = num_features
        self.num_labels = num_labels
        self.seed = seed

    def drop(self,data):
        result = data.copy()
        [m,n] = data.shape
        for i in range(m):
            for j in range(n):
                if result[i,j]==1:
                    result[i,j] = np.random.binomial(1,1-self.rate)
        return result
    
    def set_alpha(self,alpha):
        self.alpha = alpha

    def set_beta(self,beta):
        self.beta = beta

    def set_expData(self,dataset='mediamill',rate=0,evaltype='top1',margin=0.4):
        self.dataset = dataset
        self.rate = rate
        self.evaltype = evaltype
        self.margin = 0.4

    def set_trainParameters(self,epoches=50,batch_size=1024,logdir=None):
        self.epoches = epoches
        self.batch_size = batch_size
        self.logdir = logdir
        
    def init_placeholders(self):
        self.input_x = tf.placeholder(tf.float32,shape=None)
        self.input_y = tf.placeholder(tf.float32,shape=None)
        self.count = tf.placeholder(tf.float32,shape=None)
        self.phase = tf.placeholder(tf.bool,name='phase')
        
    def init_parameters(self):
        with tf.variable_scope('input_x'):
            self.w = [None]*len(self.layers)
            self.b = [None]*len(self.layers)
            for i in range(len(self.layers)):
                if i==0:
                    weights = tf.random_uniform([self.num_features,self.layers[0]],minval=-np.sqrt(6)/np.sqrt(self.num_features+self.layers[0]),maxval=np.sqrt(6)/np.sqrt(self.num_features+self.layers[0]))
                else:
                    weights = tf.random_uniform([self.layers[i-1],self.layers[i]],minval=-np.sqrt(6)/(np.sqrt(self.layers[i-1]+self.layers[i])),maxval=np.sqrt(6)/(np.sqrt(self.layers[i-1]+self.layers[i])))
                self.w[i] = tf.Variable(weights)
                self.b[i] = tf.Variable(tf.constant(0.001,shape=[self.layers[i]]))
        with tf.variable_scope('input_y'):
            self.ytabel = tf.Variable(tf.random_uniform([self.num_labels,self.layers[-1]],minval=-np.sqrt(6)/np.sqrt(self.num_labels+self.layers[-1]),maxval=np.sqrt(6)/np.sqrt(self.num_labels+self.layers[-1])))
            self.biase = tf.Variable(tf.constant(0.001,shape=[self.num_labels]))

    def build_graph(self):
            feax = [self.input_x]
            for i in range(len(self.layers)):
                tmp = tf.matmul(feax[i],self.w[i])+self.b[i]
                tmp = utils.batch_norm(tmp,self.phase)
                tmp = tf.nn.relu(tmp)
                feax.append(tmp)
            logits = tf.matmul(feax[-1],self.ytabel,transpose_b=True)+self.biase
            self.logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=logits))
            ytabel_inner = tf.matmul(self.ytabel,self.ytabel,transpose_b=True)
            y_countx = tf.multiply(self.count,(1-tf.diag(tf.ones([self.num_labels]))))
            y_count = tf.multiply(self.count,tf.diag(tf.ones([self.num_labels])))
            corrloss = -1*tf.reduce_sum(tf.multiply(ytabel_inner,y_countx))/tf.reduce_sum(y_countx)
            y_cov = tf.matmul(tf.matmul(tf.transpose(self.ytabel),y_count),self.ytabel)/tf.cast(tf.reduce_sum(y_count)-1,tf.float32)
            traceloss = 0.5*tf.reduce_sum(tf.reshape(y_cov,[-1])*tf.reshape(y_cov,[-1]))
            self.aceloss = corrloss+traceloss
            pre_expand = self.margin+tf.expand_dims(logits,axis=2)-tf.expand_dims(logits,axis=1)
            label_expand = tf.expand_dims(self.input_y,axis=2)-tf.expand_dims(self.input_y,axis=1)
            cond = (label_expand<0)
            ones = array_ops.ones_like(label_expand,dtype=label_expand.dtype)
            zeros = array_ops.zeros_like(label_expand,dtype=label_expand.dtype)
            weights = array_ops.where(cond,ones,zeros)
            weights = tf.cast(weights,dtype = pre_expand.dtype)
            self.hingeloss2 = tf.reduce_mean(tf.nn.relu(tf.multiply(pre_expand,weights)))
            self.totalloss = self.logloss+self.alpha*self.aceloss+self.beta*self.hingeloss2
            self.prediction = tf.nn.sigmoid(logits)

    def evaluation(self,pre,truth):
        if self.evaltype=='top1':
            pre[range(len(pre)),np.argmax(pre,axis=1)] = 1
            pre[pre<1] = 0
        elif self.evaltype=='threshhold':
            pre[pre>0.5] = 1
            pre[pre<=0.5] = 0
        precision = np.sum(pre*truth)/np.sum(pre)
        recall = np.sum(pre*truth)/np.sum(truth)
        fscore = 2*precision*recall/(precision+recall)
        acc = np.sum(pre==truth)/(truth.shape[0]*truth.shape[1])
        return acc,precision,fscore
    
    def start_graph(self,data_x,data_y,test_x,test_y):
        count = np.dot(data_y.T,data_y)
        self.num_samples = data_x.shape[0]
        self.num_features = data_x.shape[1]
        self.num_labels = data_y.shape[1]
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            self.init_placeholders()
            self.init_parameters()
            self.build_graph()
            self.trainer = self.optimizer.minimize(self.totalloss)
            self.init_all_vars = tf.global_variables_initializer()
            testfscore = []
            f = open(self.logdir,'a')
            f.write(time.strftime('%Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
            f.write('dataset:{} droprate:{} alpha:{} beta:{}\n'.format(self.dataset,self.rate,self.alpha,self.beta))
            with tf.Session() as sess:
                sess.run(self.init_all_vars)
                steps_per_epoch = int(self.num_samples/self.batch_size)
                for i in range(self.epoches):
                    idx = list(range(self.num_samples))
                    np.random.shuffle(idx)
                    data_x = data_x[idx]
                    data_y = data_y[idx]
                    for j in range(steps_per_epoch):
                        rand_x = data_x[j*self.batch_size:min((j+1)*self.batch_size,self.num_samples)]
                        rand_y = data_y[j*self.batch_size:min((j+1)*self.batch_size,self.num_samples)]
                        sess.run(self.trainer,feed_dict={self.input_x:rand_x,self.input_y:rand_y,self.count:count,self.phase:True})
                    #trainlossv,loglossv,acelossv = sess.run([self.totalloss,self.logloss,self.aceloss],feed_dict={self.input_x:data_x,self.input_y:data_y,self.count:count,self.phase:False})
                    #testlossv,predictionv = sess.run([self.totalloss,self.prediction],feed_dict={self.input_x:test_x,self.input_y:test_y,self.count:count,self.phase:False})
                    predictionv = sess.run(self.prediction,feed_dict={self.input_x:test_x,self.phase:False})
                    acc,precision,fscore = self.evaluation(predictionv,test_y)
                    #print('epoch{}---trainloss{},logloss{},aceloss{}'.format(i,trainlossv,loglossv,acelossv))
                    #print('epoch{}---testloss{},testacc{},testprecision{},testfscore{}'.format(i,testlossv,acc,precision,fscore))
                    #f.write('epoch{}---trainloss{},logloss{},aceloss{}\n'.format(i,trainlossv,loglossv,acelossv))
                    #f.write('epoch{}---testloss{},testacc{},testprecision{},testfscore{}\n'.format(i,testlossv,acc,precision,fscore))
                    print('epoch{}---testacc{},testprecision{},testfscore{}'.format(i,acc,precision,fscore))
                    f.write('epoch{}---testacc{},testprecision{},testfscore{}\n'.format(i,acc,precision,fscore))
                    testfscore.append(fscore)
            avgfscore = np.mean(np.array(testfscore[-10:]))
            print('average fsoce:%f'%(avgfscore))
            f.write('average fsoce:%f\n'%(avgfscore))


                
