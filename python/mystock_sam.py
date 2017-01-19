# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:50:47 2017

@author: sunlei
"""


from sklearn.externals import joblib
from mmmall import MMModel
import tushare as ts
import pandas as pd
import numpy as np
import os
import time
import math

class my_model:
    def __init__(self,name='mmm',path=''):
        self.name=name
        self.path=path
        self.fullpath=self.path+self.name+".pkl"
        self.indim=20
    def load(self):
        self.clf=joblib.load(self.fullpath)
        self.indim=self.clf.indim
    def predict(self,X):
        sl=self.clf.predict(X)
        return sl

mymod=my_model(name='sl',path="..\\")
mymod.load()
money=100000.00
have_stocks={}
hc={}
max_stock_num=5
sl=ts.get_industry_classified()
print 'end1'
stocks_all=set(sl['code'])
sl=ts.get_gem_classified()
print '获取创业板'
stocks_gam=set(sl['code'])
sl=ts.get_st_classified()
print '获取st'
stocks_st=set(sl['code'])
stocks_list=stocks_all-stocks_gam
stocks_list=stocks_list-stocks_st
jbzl=ts.get_stock_basics()
print '获取基本信息'
stocks_total_gb=jbzl['totals']
szzs=ts.get_hist_data('399106')
open_days=list(szzs.index)
open_days.sort()
zs_open= szzs.pop('open')
zs_close= szzs.pop('close')
zs_pch= szzs.pop('p_change')
pchs = pd.DataFrame(zs_pch,columns=['p_change'])
pchs.columns=['zs']
closes = pd.DataFrame(zs_close,columns=['close'])
closes.columns=['zs']
opens = pd.DataFrame(zs_open,columns=['open'])
opens.columns=['zs']
opens.sort_index(inplace=True)
closes.sort_index(inplace=True)
pchs.sort_index(inplace=True)
a=0
l=len(stocks_list)
if os.path.exists('sl.hdf'):
    pchs=pd.read_hdf('sl.hdf','p_change')
    opens=pd.read_hdf('sl.hdf','open')
    closes=pd.read_hdf('sl.hdf','close')
else:
    for stock in stocks_list:
        df=ts.get_hist_data(stock)
        a=a+1
        print a,l
        if df is None:continue
        s1=df.pop('open')
        s2=df.pop('close')
        s3=df.pop('p_change')
        opens[stock]=s1
        closes[stock]=s2
        pchs[stock]=s3
stocks_list=list(opens.T.index)
tc_num=10
day_count=0
drop_day=5
drop_rate=0.05
for i in range(open_days.index('2016-01-04'),len(open_days)):
    df=closes['zs'].dropna()
    close1=df.loc[open_days[i-drop_day]]
    close2=df.loc[open_days[i-1]]
    print open_days[i-drop_day],open_days[i-1],close1,close2,close2/close1
    if (1-close2/close1)>drop_rate:
        if len(have_stocks)>0:
            hs_keys=have_stocks.keys()
            hs_set=set(hs_keys)
            s1=0.0
            for stock in hs_set:
                s1=have_stocks[stock][2]
                money=money+s1
                have_stocks.pop(stock)
        day_count=0
        print '清仓',money
        continue
    if day_count % tc_num != 0: 
        day_count += 1
        if len(have_stocks)>0:
            s1=money
            for stock in have_stocks.keys():
                try:
                    cl=closes.loc[open_days[i]].loc[stock]
                except Exception,e:
                    cl=float('nan')
                if not math.isnan(cl):
                    have_stocks[stock][2]=have_stocks[stock][0]*closes.loc[open_days[i]].loc[stock]
                s1=s1+have_stocks[stock][2]
        print day_count,open_days[i],s1,money,have_stocks
        continue
    s_time=time.time()
    start_day=open_days[i-mymod.indim]
    end_day=open_days[i]
    test=[]
    stock_pr={}
    totl=0.0
    stocks_total=pd.DataFrame()
    for stock in stocks_list:
        ch=0
        df=opens[stock].dropna()
        try:
            st_days=list(df.index)
            end_d=st_days.index(end_day)
            start_d=st_days.index(start_day)
        except Exception,e:
            continue
        if end_d-start_d!=mymod.indim:
            continue
       
        stock_data = closes[stock].dropna()
        close_prices = list(pchs[stock].dropna().values)[start_d:end_d]
        close_prices=[x/10.1 for x in close_prices]
        #if len(close_prices)!=mymod.indim:continue
        close_price = stock_data.iloc[end_d-1]
        stock_pr[stock]=[df.iloc[end_d],stock_data.iloc[end_d]]
        ch = mymod.predict(np.array(close_prices).reshape((1,-1)))
        try:
            st_tol=stocks_total_gb.loc[stock]
        except Exception,e:
            continue
        
        if ch==1 and st_tol>0:
            test.append(stock)
            stocks_total[stock]=[stocks_total_gb.loc[stock]*close_price]
    if len(test)==0:continue
    print test
    sp=stocks_total.T
    sp.columns=['A']
    ch_stocks=list(sp.sort_values(by='A').index)
    if len(ch_stocks)>max_stock_num:
        choice=ch_stocks[:max_stock_num]
    else:
        choice=ch_stocks
    ch_num=len(choice)
    times1= time.time()-s_time
    print times1
    if ch_num<1:continue
    if len(have_stocks)>0:
        hs_keys=have_stocks.keys()
        hs_set=set(hs_keys)
        ch_st=set(choice)
        sell_ls=hs_set-ch_st
        for stock in hs_set:
            try:
                s1=have_stocks[stock][0]*stock_pr[stock][0]
            except Exception,e:
                s1=have_stocks[stock][2]
            money=money+s1
            have_stocks.pop(stock)
    buy_money=money/ch_num
    for stock in choice:
        s1=int(buy_money/stock_pr[stock][0]/100)*100
        have_stocks[stock]=[s1,stock_pr[stock][0],s1*stock_pr[stock][1]]
        money=money-s1*stock_pr[stock][0]
        totl=totl+have_stocks[stock][2]
    hc[end_day]=totl+money
    day_count += 1
    print day_count,hc

        
        
    
        
        