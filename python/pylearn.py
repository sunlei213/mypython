# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:01:49 2016

@author: sunlei
"""
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.structure.modules import *
import tushare as ts
import numpy as np

HISTORY=20
label1=10
universe=['600030']
training_set = ("2013-04-01", "2016-07-31")       # 训练集（六年）
testing_set  = ("2016-01-15", "2016-03-22")       # 测试集（2015上半年数据）

def make_data(date_set):
    ds = SupervisedDataSet(HISTORY, 1)
    for ticker in universe: # 遍历每支股票
        raw_data = ts.get_hist_data(ticker, start=date_set[0], end=date_set[1])
        plist = list(raw_data['close'])
        buy_prices = list(raw_data['open'])
        high_prices = list(raw_data['high'])
        low_prices = list(raw_data['low'])
        for idx in range(1, len(plist) - HISTORY -label1 - 1):
            sample = []
            for i in range(HISTORY):
                sample.append(plist[idx + i] / plist[idx + i - 1] - 1)
            buy_price = buy_prices[idx + HISTORY]
            high_price = max( high_prices[idx + HISTORY:idx + HISTORY+label1])
            low_price = min( low_prices[idx + HISTORY:idx + HISTORY+label1])
            if (high_price/buy_price-1)>0.1:
                answer = 1
            else:
                answer =-1

            ds.addSample(sample, answer)
    return ds
### 建立测试集
def make_testing_data():
    ds = SupervisedDataSet(HISTORY, 1)
    for ticker in universe: # 遍历每支股票
        raw_data =  ts.get_hist_data(ticker, start=testing_set[0], end=testing_set[1])
        plist = list(raw_data['close'])
        for idx in range(1, len(plist) - HISTORY - 1):
            sample = []
            for i in range(HISTORY):
                sample.append(plist[idx + i - 1] / plist[idx + i] - 1)
            answer = plist[idx + HISTORY - 1] / plist[idx + HISTORY] - 1

            ds.addSample(sample, answer)
    return ds
def random_data(dataset):
    ds=SupervisedDataSet(dataset['input'].shape[1], 1)
    ds.clear()
    for i in np.random.permutation(len(dataset)):
        ds.addSample(dataset['input'][i],dataset['target'][i])
    return ds
def save_arguments(net):
    NetworkWriter.writeToFile(net, 'huge_data.csv')
    print 'Arguments save to file net.csv'

### 构造BP训练实例
def make_trainer(net, ds, learningrate=0.01,momentum = 0.1, verbose = True, weightdecay = 0.01): # 网络, 训练集, 训练参数
    trainer = BackpropTrainer(net, ds, learningrate=learningrate,momentum = momentum, verbose = verbose, weightdecay = weightdecay)
    return trainer
### 开始训练
def start_training(trainer, epochs = 15): # 迭代次数
    trainer.trainEpochs(epochs)

def start_testing(net, dataset):
    return net.activateOnDataset(dataset)

### 初始化神经网络
fnn = buildNetwork(HISTORY, 200, 20,2, 1, hiddenclass=TanhLayer,outclass=SoftmaxLayer)
ds=random_data(make_data(training_set))
training_dataset,testing_dataset = ds.splitWithProportion(0.9)
testing_dataset  = random_data(make_data(testing_set))
trainer = make_trainer(fnn, training_dataset,0.005)
start_training(trainer,50)
save_arguments(fnn)
s1=start_testing(fnn, testing_dataset )
for i in range(len(s1)):
    print s1[i],testing_dataset['target'][i]

tstresult = percentError( s1, testing_dataset['target'] )
print tstresult