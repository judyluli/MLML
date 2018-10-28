# test 
import acev1
import input
model = acev1.ACE(rate=0.2)
model.set_trainParameters(logdir='/home/lulusee/MLML/mediamill/log.txt')
model.start_graph()
data_x,data_y = input.readfile('/data/users/lulusee/mediamill/mediamill-train.arff')
test_x,test_y = input.readfile('/data/users/lulusee/mediamill/mediamill-test.arff')
model.start_graph(data_x,data_y,test_x,test_y)