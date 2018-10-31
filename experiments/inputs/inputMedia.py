import numpy as np
def readfile(path):
    x=[]
    y=[]
    f=open(path)
    while(True):
        line = f.readline().strip()
        if(line.startswith('@data')):
            print('breakline:'+line)
            break
    line = f.readline().strip()
    while(line):
        data = [float(x) for x in line.split(',')]
        x.append(data[:120])
        y.append(data[120:])
        line = f.readline().strip()
    return np.array(x),np.array(y)
'''
x,y = readfile('/data/users/lulusee/mediamill/mediamill-test.arff')
print(x.shape)
print(y.shape)
'''
