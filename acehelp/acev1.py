import tensorflow as tf
import utils
import numpy as np
import time
# this is test for github
class ACE():
    def __init__(self,layers=[50,50,50,50],alpha=0.001,rate=0,optimizer=tf.train.AdamOptimizer(0.001),num_features=None,num_labels=None,seed=None):
        self.layers = layers
        self.alpha = alpha
        self.rate = rate
        self.optimizer = optimizer
        self.num_features = num_features
        self.num_labels = num_labels
        self.seed = seed
    
    def set_trainParameters(self,epoches=50,batch_size=1024,logdir=None):
        self.epoches = epoches
        self.batch_size = batch_size
        self.logdir = logdir
        
    def init_placeholders(self):
        self.input_x = tf.placeholder(tf.float32,shape=None)
        self.input_y = tf.placeholder(tf.float32,shape=None)
        self.count = tf.placeholder(tf.float32,shape=None)
        self.phase = tf.placeholder(tf.bool,name='phase')
        
    def drop(self,data):
        result = data.copy()
        [m,n] = data.shape
        for i in range(m):
            for j in range(n):
                if result[i,j]==1:
                    result[i,j] = np.random.binomial(1,1-self.rate)
        return result
        
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
            self.feax = [self.input_x]
            for i in range(len(self.layers)):
                tmp = tf.matmul(self.feax[i],self.w[i])+self.b[i]
                tmp = utils.batch_norm(tmp,self.phase)
                tmp = tf.nn.relu(tmp)
                self.feax.append(tmp)
            self.logits = tf.matmul(self.feax[-1],self.ytabel,transpose_b=True)+self.biase
            self.logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=self.logits))
            self.ytabel_inner = tf.matmul(self.ytabel,self.ytabel,transpose_b=True)
            self.y_countx = tf.multiply(self.count,(1-tf.diag(tf.ones([self.num_labels]))))
            self.y_count = tf.multiply(self.count,tf.diag(tf.ones([self.num_labels])))
            self.corrloss = -1*tf.reduce_sum(tf.multiply(self.ytabel_inner,self.y_countx))/tf.reduce_sum(self.y_countx)
            self.y_cov = tf.matmul(tf.matmul(tf.transpose(self.ytabel),self.y_count),self.ytabel)/tf.cast(tf.reduce_sum(self.y_count)-1,tf.float32)
            self.traceloss = 0.5*tf.reduce_sum(tf.reshape(self.y_cov,[-1])*tf.reshape(self.y_cov,[-1]))
            self.aceloss = self.corrloss+self.traceloss
            self.totalloss = (1-self.alpha)*self.logloss+self.alpha*self.aceloss
            self.prediction = tf.nn.sigmoid(self.logits)

    def evaluation(self,pre,truth):
        pre[range(len(pre)),np.argmax(pre,axis=1)] = 1
        pre[pre<1] = 0
        precision = np.sum(pre*truth)/np.sum(pre)
        recall = np.sum(pre*truth)/np.sum(truth)
        fscore = 2*precision*recall/(precision+recall)
        acc = np.sum(pre==truth)/(truth.shape[0]*truth.shape[1])
        return acc,precision,fscore
    
    def start_graph(self,data_x,data_y,test_x,test_y):
        data_y = self.drop(data_y)
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
            f = open(self.logdir,'w')
            f.write(time.strftime('%Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
            f.write('alpha:{} droprate:{}\n'.format(self.alpha,self.rate))
            with tf.Session() as sess:
                sess.run(self.init_all_vars)
                steps_per_epoch = int(self.num_samples/self.batch_size)
                for i in range(self.epoches):
                    idx = list(range(len(data_x)))
                    np.random.shuffle(idx)
                    data_x = data_x[idx]
                    data_y = data_y[idx]
                    for j in range(steps_per_epoch):
                        rand_x = data_x[j*self.batch_size:min((j+1)*self.batch_size,self.num_samples)]
                        rand_y = data_y[j*self.batch_size:min((j+1)*self.batch_size,self.num_samples)]
                        sess.run(self.trainer,feed_dict={self.input_x:rand_x,self.input_y:rand_y,self.count:count,self.phase:True})
                    trainlossv,loglossv,acelossv = sess.run([self.totalloss,self.logloss,self.aceloss],feed_dict={self.input_x:data_x,self.input_y:data_y,self.count:count,self.phase:False})
                    testlossv,predictionv = sess.run([self.totalloss,self.prediction],feed_dict={self.input_x:test_x,self.input_y:test_y,self.count:count,self.phase:False})
                    acc,precision,fscore = self.evaluation(predictionv,test_y)
                    print('epoch{}---trainloss{},logloss{},aceloss{}'.format(i,trainlossv,loglossv,acelossv))
                    print('epoch{}---testloss{},testacc{},testprecision{},testfscore{}'.format(i,testlossv,acc,precision,fscore))
                    f.write('epoch{}---trainloss{},logloss{},aceloss{}\n'.format(i,trainlossv,loglossv,acelossv))
                    f.write('epoch{}---testloss{},testacc{},testprecision{},testfscore{}\n'.format(i,testlossv,acc,precision,fscore))
                    testfscore.append(fscore)
            avgfscore = np.mean(np.array(testfscore[-10:]))
            print('average fsoce:%f'%(avgfscore))
            f.write('average fsoce:%f\n'%(avgfscore))


                
