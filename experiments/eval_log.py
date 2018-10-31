import sys
path = sys.argv[1]

f = open(path)
fw = open('./excel.txt','a')
dataset = None
alpha = None
beta = None
droprate = None
lines = f.readlines()
'''
fw.write('alpha {} {} {} {} {} {} '.format(0,0.0001,0.0005,0.001,0.002,0.005))
for line in lines:
    if line.startswith('dataset'):
        curdataset,curalpha,curdroprate = line.strip().split()
        if curdataset!=dataset:
            fw.write('\n'+curdataset)
            dataset = curdataset
        if curdroprate!=droprate:
            fw.write('\n'+curdroprate+' ')
            droprate = curdroprate
    if line.startswith('average'):
            fw.write(line.strip().split(':')[1]+'  ')
'''
fw.write('beta {} {} {} {} '.format(0,0.01,0.1,0.5))
for line in lines:
    if line.startswith('dataset'):
        curdataset,curdroprate,curalpha,curbeta = line.strip().split()
        if curdataset!=dataset:
            fw.write('\n'+curdataset)
            dataset = curdataset
        if curdroprate!=droprate:
            fw.write('\n'+curdroprate)
            droprate = curdroprate
        if curalpha!=alpha:
            fw.write('\n'+curalpha+' ')
            alpha = curalpha
    if line.startswith('average'):
            fw.write(line.strip().split(':')[1]+' ')

