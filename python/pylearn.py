# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:01:49 2016

@author: sunlei
"""
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter,NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.structure.modules import *
import tushare as ts
import numpy as np
import time

HISTORY=20
label1=5

training_set = ("2015-02-01", "2017-03-17")       # 训练集（六年）
testing_set  = ("2016-01-05", "2016-12-31")       # 测试集（2015上半年数据）

def sigmoid(X,useStatus=True):  
    if useStatus:  
        return 1.0 / (1 + np.exp(-float(X)));  
    else:  
        return float(X);  
               
def make_data(date_set):
    ds = SupervisedDataSet(HISTORY, 2)
    for ticker in universe: # 遍历每支股票
        df = ts.get_k_data(ticker, start=date_set[0], end=date_set[1])
        if str(type(df))!="<class 'NoneType'>":
            raw_data = df.dropna()
            plist = list(raw_data['close'])
            buy_prices = list(raw_data['open'])
            high_prices = list(raw_data['high'])
            low_prices = list(raw_data['low'])
            for idx in range(1, len(plist) - HISTORY -label1 - 1):
                sample = []
                for i in range(HISTORY):
                    sample.append((plist[idx + i] / plist[idx + i - 1] - 1)*9)
                buy_price = buy_prices[idx + HISTORY]
                high_price = max( high_prices[idx + HISTORY:idx + HISTORY+label1])
                low_price = min( low_prices[idx + HISTORY:idx + HISTORY+label1])
                #answer=high_price/buy_price-1
                if (high_price/buy_price-1)>0.05:
                    answer = [1,0]
                else:
                    answer = [0,1]
    
                ds.addSample(sample, answer)
    return ds
### 建立测试集
def make_testing_data():
    ds = SupervisedDataSet(HISTORY, 1)
    for ticker in universe: # 遍历每支股票
        raw_data =  ts.get_k_data(ticker, start=testing_set[0], end=testing_set[1])
        plist = list(raw_data['close'])
        for idx in range(1, len(plist) - HISTORY - 1):
            sample = []
            for i in range(HISTORY):
                sample.append(plist[idx + i - 1] / plist[idx + i] - 1)
            answer = plist[idx + HISTORY - 1] / plist[idx + HISTORY] - 1

            ds.addSample(sample, answer)
    return ds
def random_data(dataset):
    ds=SupervisedDataSet(dataset['input'].shape[1], 2)
    ds.clear()
    for i in np.random.permutation(len(dataset)):
        ds.addSample(dataset['input'][i],dataset['target'][i])
    return ds
def save_arguments(net,filename):
    NetworkWriter.writeToFile(net, filename)
    print 'Arguments save to file net.csv'

def load_arguments(filename):
    return NetworkReader.readFrom(filename)

### 构造BP训练实例
def make_trainer(net, ds, learningrate=0.01,momentum = 0.1, verbose = True, weightdecay = 0.1): # 网络, 训练集, 训练参数
    trainer = BackpropTrainer(net, ds, learningrate=learningrate,momentum = momentum, verbose = verbose, weightdecay = weightdecay)
    return trainer
### 开始训练
def start_training(trainer, epochs = 15): # 迭代次数
    trainer.trainEpochs(epochs)

def start_testing(net, dataset):
    return net.activateOnDataset(dataset)

### 初始化神经网络

#sl=ts.get_today_all()
#sl=list((sl.set_index('code'))['mktcap'].sort_values().index[:400])
sl=ts.get_zz500s()
sl=list(sl['code'])
universe=sl

#universe=['600030']
ds=random_data(make_data(training_set))
training_dataset,testing_dataset = ds.splitWithProportion(0.9)
#training_dataset = ds
print(len(ds))
fnn = load_arguments('500_1.xml')
#fnn = buildNetwork(HISTORY, 200, 100,50, 2, bias = True,recurrent=True, hiddenclass=TanhLayer,outclass=SoftmaxLayer )
#testing_dataset  = random_data(make_data(testing_set))
trainer = make_trainer(fnn, training_dataset,0.005,weightdecay = 0.00001)
s_time=time.time()
print s_time
for i in range(100):
    b_time=time.time()
    start_training(trainer,5)
    print time.time()-b_time

#mmm.fit(X_train,y_train)
run_time=time.time()-s_time
print run_time

save_arguments(fnn,'500_1_1.xml')
s1=start_testing(fnn, testing_dataset )
t=[0]*10
f=[0]*10
for i in range(len(s1)):
#    print s1[i],testing_dataset['target'][i]
    x = s1[i][0]-s1[i][1]
    for l in range(10):
        if x>float(0.1*l):
            if testing_dataset['target'][i][0]==1:
                t[l] +=1
            else:
                f[l] +=1
for l in range(10):
    print '>0.{0}:命中率：{1:.2f}% ,命中数：{2} ,总数：{3} '.format(l,float(t[l])*100/float(t[l]+f[l]),t[l],t[l]+f[l])

tstresult = percentError( s1, testing_dataset['target'] )
print tstresult
