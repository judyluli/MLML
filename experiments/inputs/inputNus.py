import numpy as np
import os
def readx(predex,flag):
    data = None
    if flag=='Train':
        pathlist=['Train_Normalized_CORR.dat','Train_Normalized_CH.dat','Train_Normalized_CM55.dat','Train_Normalized_EDH.dat','Train_Normalized_WT.dat','BoW_Train_int.dat']
    else:
        pathlist=['Test_Normalized_CORR.dat','Test_Normalized_CH.dat','Test_Normalized_CM55.dat','Test_Normalized_EDH.dat','Test_Normalized_WT.dat','BoW_Test_int.dat']
    for i in pathlist:
        path = os.path.join(predex,i)
        f = open(path)
        tmp = [list(map(float,x.strip().split())) for x in f.readlines()]
        tmp = np.array(tmp)
        if data is None:
            data = tmp
        else:
            data = np.concatenate((data,tmp),axis=1)
    return data
def ready(path):
    f = open(path)
    tmp = [list(map(float,x.strip().split())) for x in f.readlines()]
    tmp = np.array(tmp)
    return tmp
