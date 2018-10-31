# test 
import sys
sys.path.append("..")
from acehelp import acev1
from inputs import inputMedia
from inputs import inputNus
import tensorflow as tf

data_x,data_y = inputMedia.readfile('/data/users/lulusee/mediamill/mediamill-train.arff')
test_x,test_y = inputMedia.readfile('/data/users/lulusee/mediamill/mediamill-test.arff')
model = acev1.ACE()
for rate in [0,0.1,0.2,0.3,0.4]:
    model.set_expData(dataset='mediamill',rate=rate,evaltype='threshhold')
    drop_y = model.drop(data_y)
    for alpha in [0,0.0001,0.0005,0.001,0.002,0.005]:
        model.set_alpha(alpha=alpha)
        model.set_trainParameters(epoches=50,logdir='/home/lulusee/MLML/experiments/log.txt')
        model.start_graph(data_x,drop_y,test_x,test_y)

data_x = inputNus.readx('/data/users/lulusee/nus/Low_Level_Features','Train')
data_y = inputNus.ready('/data/users/lulusee/nus/Tags/Train_Tags1k.dat')
test_x = inputNus.readx('/data/users/lulusee/nus/Low_Level_Features','Test')
test_y = inputNus.ready('/data/users/lulusee/nus/Tags/Test_Tags1k.dat')
model = acev1.ACE()
for rate in [0,0.1,0.2,0.3,0.4]:
    model.set_expData(dataset='nuswide',rate=rate,evaltype='top1')
    drop_y = model.drop(data_y)
    for alpha in [0,0.0001,0.0005,0.001,0.002,0.005]:
        model.set_alpha(alpha=alpha)
        model.set_trainParameters(epoches=50,logdir='/home/lulusee/MLML/experiments/log.txt')
        model.start_graph(data_x,drop_y,test_x,test_y)
