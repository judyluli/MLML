import tensorflow as tf
import utils
# this is test for github
class ACE():
    def
    __init__(self,layers=[50,50,50,50],alpha=0.001,optimizer=tf.train.AdamOptimizer(0.001),num_features=None,num_labels=None,seed=None):
        self.layers = layers
        self.alpha = alpha
        self.optimizer = optimizer
        self.num_features = num_features
        self.num_labels = num_labels
        self.seed = seed
    def init_placeholders(self):
        self.input_x = tf.placeholder(tf,float32,shape=None)
        self.input_y = tf.placeholder(tf.float32,shape=None)
        self.count = tf.placeholder(tf.float32,shape=None)
        self.phase = tf.placeholder(tf.bool,name='phase')
    def drop(self,data,rate):
        result = data.copy()
        [m,n] = data.shape
        for i in range(m):
            for j in range(n):
                if result[i,j]==1:
                    result[i,j] = np.random.binomial(1,1-rate)
        return result
    def init_parameters(self):
        with tf.variabel_scope('input_x'):
            self.w = [None]*len(self.layers)
            self.b = [None]*len(self.layers)
            for i in range(len(self.layers)):
                if i==0:
                    weights = tf.random_uniform([self.num_features,self.layers[0]],minval=-np.sqrt(6)/np.sqrt(self.num_features+self.layers[0]),maxval=np.sqrt(6)/np.sqrt(self.num_features+self.layers[0]))
                else:
                    weights = tf.random_uniform([self.layers[i-1],self.layers[i]],minval=-np.sqrt(6)/(np.sqrt(self.layers[i-1]+self.layers[i])),maxval=np.sqrt(6)/(np.sqrt(self.layers[i-1]+self.layers[i])))
                self.w[i] = tf.Variable(weights)
                self.b[i] = tf.Variabel(tf.constant(0.001,shape=[self.layers[i]]))
        with tf.variabel_scope('input_y'):
            self.yatbel = tf.Variable(tf.random_uniform([self.num_labels,self.layers[-1]],minval=-np.sqrt(6)/np.sqrt(self.num_labels+self.layers[-1]),maxval=np.sqrt(6)/np.sqrt(self.num_labels+self.layers[-1]))
            self.biase = tf.Variabel(tf.constant(0.001,shape=[self.num_labels]))

    def build_graph(self):
            self.feax = [self.input_x]
            for i in range(len(self.layers)):
                tmp = tf.matmul(self.feax[i],self.w[i])+self.b[i]
                tmp = utils.batch_norm(tmp,self.phase)
                tmp = tf.nn.relu(tmp)
                self.feax.append(tmp)
            self.logits = tf.matmul(self.feax[-1],self.ytabel,transpose_b=True)+self.biase
            self.logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=self.logits))
            self.ytable_inner = tf.matmul(self.ytabel,self.ytabel,transpose_b=True)
            self.y_countx = tf.multiply(self.count,(1-tf.diag(tf.ones([self.num_labels]))))
            self.y_count = tf.multiply(self.count,tf.diag(tf.ones([self.num_labels])))
            self.corrloss = -1*tf.reduce_sum(tf.multiply(self.ytabel_inner,self.y_countx))/tf.reduce_sum(self.y_countx)
            self.y_cov = tf.matmul(tf.matmul(tf.transpose(self.ytable),self.y_count),self.ytabel)/tf.cast(tf.reduce_sum(self.y_count)-1,tf.float32)
            self.traceloss = 0.5*tf.reduce_sum(tf.reshape(self.y_cov,[-1])*tf.reshape(self.y_cov,[-1]))
            self.aceloss = self.corrloss+self.traceloss
            self.totalloss = (1-self.alpha)*self.loss+self.alpha*self.aceloss
    def start_graph(self):
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            self.init_placeholders()
            self.init_parameters()
            self.build_graph()
        self.trainer = self.optimizer.minimize(self.totalloss)
        self.init_all_vars = tf.global_variables_initializer()
