# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:04:31 2017

@author: sunlei
"""


import pandas as pd
import numpy as np
import tushare as ts
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import cPickle as pickle


class TrainDataGenerator:
    def __init__(self, name="train_samples", need_target=True, train_size_perc=0.8,indim=20,days=10):
        self.samples=None
        self.path="data/TrainPool/"
        self.fullpath=self.path+name+".csv"
        self.target_columns=['open','low','high']
        self.indim=indim
        self.outdim=days
        self.need_target = need_target
        self.train_size_perc = train_size_perc
        self.high_threshhold=0.1
        self.down_threshhold=-0.05
        self.columns=['open','low','high']
        for i in range(indim):self.columns.append(i)
    def setdata(self,stock,date_set,coun=300):
        df = ts.get_hist_data(stock, start=date_set[0], end=date_set[1])
        stock_data = df.dropna()
        close_prices = stock_data['p_change'].values
        coun=len(close_prices)
        X = []
        for index in range (self.indim,coun+1-self.outdim):
            buy_price = stock_data['open'].values[index]
            high_prices = stock_data['high'].values[index:index+self.outdim]
            low_prices = stock_data['low'].values[index:index+self.outdim]
            low_price= min(low_prices)
            high_price= max(high_prices)
            features = [buy_price,low_price,high_price]
            for day in range(self.indim):
                up_rate=close_prices[index-self.indim+day]/10
                if up_rate!=up_rate:up_rate=0.0
                features.append(up_rate)
            X.append(features)
        self.samples = pd.DataFrame(data=X, columns=self.columns)
    def setdatas(self,stock_list,date_set,coun=400):
        X = []
        a=0.0
        l=len(stock_list)
        print a,l
        for stock in stock_list:
            df=ts.get_hist_data(stock, start=date_set[0], end=date_set[1])
            stock_data = df.dropna()
            close_prices = stock_data['p_change'].values
            coun=len(close_prices)
            for index in range (self.indim,coun+1-self.outdim):
                buy_price = stock_data['open'].values[index]
                high_price = max(stock_data['close'].values[index:index+self.outdim])
                low_price = min(stock_data['close'].values[index:index+self.outdim])
                features = [buy_price,low_price,high_price]
                for day in range(self.indim):
                    up_rate=close_prices[index-self.indim+day]/10
                    if up_rate!=up_rate:up_rate=0.0
                    features.append(up_rate)
                X.append(features)
            a=a+1
            print a,float(a)/l
        self.samples = pd.DataFrame(data=X, columns=self.columns)
    def gen(self):
        self.permutation()
        if(self.need_target):
            self.split_XY()
            self.gen_y()
        else:
            self.X=self.samples.copy()
        if(self.train_size_perc < 1):
            self.split_test()
    def permutation(self):
        self.samples=self.samples.iloc[np.random.permutation(len(self.samples))]
    def split_XY(self):
        self.Y= self.samples[self.target_columns]
        self.X= self.samples.copy()
        for c in self.target_columns:
            self.X.pop(c)
        return self.X, self.Y
    def gen_y(self):
        high_score=(self.Y['high']-self.Y['open'])/self.Y['open']
        down_score=(self.Y['low']-self.Y['open'])/self.Y['open']
        pos_mask=(high_score>self.high_threshhold)&(down_score>self.down_threshhold)
        #pos_mask1=(down_score<self.down_threshhold)
        self.y=pd.Series(-np.ones(self.Y.shape[0]), index=pos_mask.index)
        self.y[pos_mask]=1
        #self.y[pos_mask1]=-1
        return self.y
    def split_test(self):
        train_size = np.int16(np.round(self.train_size_perc * self.X.shape[0]))
        self.X_train, self.y_train = self.X.iloc[:train_size, :], self.y.iloc[:train_size]
        self.X_test, self.y_test = self.X.iloc[train_size:, :], self.y.iloc[train_size:]
class MMModel:
    def __init__(self, name='mmm', path='data/TrainPool/',n_pca='mle', C_svr=1.0):
        self.n_pca=n_pca
        self.C_svr=C_svr
        self.name=name
        self.path=path
        self.fullpath= self.path+self.name+".pkl"
    def save(self):
        pickle_file = open('..\\indim.pkl', 'wb')
        pickle.dump(self.indim, pickle_file)
        pickle_file.close()
        pickle_file = open('..\\mod_norm.pkl', 'wb')
        pickle.dump(self.mod_norm, pickle_file)
        pickle_file.close()
        pickle_file = open('..\\mod_demR.pkl', 'wb')
        pickle.dump(self.mod_demR, pickle_file)
        pickle_file.close()
        pickle_file = open('..\\mod_train.pkl', 'wb')
        pickle.dump(self.mod_train, pickle_file)
        pickle_file.close()
        return

    def load(self,str1):
        return pickle.loads(str1)
        
    def fit(self,X,y):
        self.mod_norm=StandardScaler()
        Xtrans = self.mod_norm.fit_transform(X)
        self.indim=Xtrans.shape[1]
        self.mod_demR=PCA(n_components=self.n_pca, svd_solver='full')
        Xtrans = self.mod_demR.fit_transform(Xtrans)
        print self.indim
        self.mod_train=SVC(kernel='rbf', C=self.C_svr)
        w,weight=self.gen_svr_w(y)
        if(weight<1 or weight>40):
            print("unbalance sample: " + weight)
        self.mod_train.fit(Xtrans,y,w)
    def gen_svr_w(self,y):
        tol=y.shape[0]
        pos=y[y==1].shape[0]
        neg=tol-pos
        
        w=pd.Series(np.ones(y.shape[0]), y.index)
        if(pos==0 or neg==0):
            return w,0
        if(pos<neg):
            weight=float(neg)/pos
            w[y==1]=weight
        else:
            weight=float(pos)/neg
            w[y==-1]=weight
        return w,weight
    def transform(self,X,y=None):
        Xtrans = self.mod_norm.transform(X)
        Xtrans = self.mod_demR.transform(Xtrans)
        return Xtrans
    def predict(self,X,y=None):
        Xtrans =self.transform(X)
        return self.mod_train.predict(Xtrans)
    def report(self,X,y=None):
        Xtrans =self.transform(X)
        p=self.mod_train.predict(Xtrans)
        d=self.mod_train.decision_function(Xtrans)
        return pd.DataFrame({'predict':p, 'dec_func':d})
    def score(self,X,y=None):
        Xtrans =self.transform(X)
        return self.mod_train.score(Xtrans,y), metrics.f1_score(y,self.predict(X))
