# test 
import sys
sys.path.append("..")
from acehelp import acev1
from inputs import inputMedia
from inputs import inputNus
import tensorflow as tf

rate = tf.app.flags.DEFINE_float('rate',0,'the rate of random missing labels')
evaltype = tf.app.flags.DEFINE_string('evaltype','top1','the evaluation type :top1 or threshhold')
epoches = tf.app.flags.DEFINE_integer('epoches',10,'traing epoches')
alpha = tf.app.flags.DEFINE_float('alpha',0.001,'weight of aceloss')
dataset = tf.app.flags.DEFINE_string('dataset','mediamill','dataset name, meidamill or nuswide')

FLAGS = tf.app.flags.FLAGS
if FLAGS.dataset=='mediamill':
    data_x,data_y = inputMedia.readfile('/data/users/lulusee/mediamill/mediamill-train.arff')
    test_x,test_y = inputMedia.readfile('/data/users/lulusee/mediamill/mediamill-test.arff')
elif FLAGS.dataset=='nuswide':
    data_x = inputNus.readx('/data/users/lulusee/nus/Low_Level_Features','Train')
    data_y = inputNus.ready('/data/users/lulusee/nus/Tags/Train_Tags1k.dat')
    test_x = inputNus.readx('/data/users/lulusee/nus/Low_Level_Features','Test')
    test_y = inputNus.ready('/data/users/lulusee/nus/Tags/Test_Tags1k.dat')


model = acev1.ACE()
model.set_alpha(alpha=FLAGS.alpha)
model.set_expData(dataset=FLAGS.dataset,rate=FLAGS.rate,evaltype=FLAGS.evaltype)
model.set_trainParameters(epoches=FLAGS.epoches,logdir='/home/lulusee/MLML/experiments/log.txt')
drop_y = model.drop(data_y)
model.start_graph(data_x,drop_y,test_x,test_y)
