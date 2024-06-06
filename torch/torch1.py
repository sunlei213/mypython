# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:00:13 2024

@author: sunlei
"""

######################################导入一堆乱七八糟的库###############################################

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels import regression
from six import StringIO
#导入pca
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
import seaborn as sns
import sqlite3
from pandas.io import sql
from datetime import datetime, timedelta

def get_price(stock, start, end, con):
    sql_st = "select distinct * from '{0}' where date between '{1}' and '{2}' order by date".format(
        stock, start, end)
    df = sql.read_sql(sql_st, con, index_col=['date'])
    return df

#获取指定周期的日期列表 'W、M、Q'
def get_period_date(peroid,start_date, end_date, con):
    #设定转换周期period_type  转换为周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'
    stock_data = get_price('510050.XSHG',start_date,end_date,con)
    #记录每个周期中最后一个交易日
    #stock_data['date']=stock_data.index
    #进行转换，周线的每个变量都等于那一周中最后一个交易日的变量值
    #period_stock_data=stock_data.resample(peroid,how='last')
    #period_stock_data = period_stock_data.set_index('date').dropna()
    stock_data.index = pd.to_datetime(stock_data.index)
    date=stock_data.index
    pydate_array = date.to_pydatetime()
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array )

    date_only_series = pd.Series(date_only_array)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date=start_date-timedelta(days=1)
    start_date = start_date.strftime("%Y-%m-%d")
    date_list=date_only_series.values.tolist()
    #date_list.insert(0,start_date)
    return date_list
#去除上市距beginDate不足3个月的ETF
def delect_stop(stocklist,beginDate,n=30*3):
    stocks = stocklist.index.tolist()
    stockList=[]
    beginDate = datetime.strptime(beginDate, "%Y-%m-%d")
    for stock in stocks:
        start_date=stocklist.loc[stock,'start_date']
        if start_date<(beginDate-timedelta(days=n)).strftime('%Y-%m-%d'):
            stockList.append(stock)
    return stockList

def get_stock(begin_date,con):
    stockList = sql.read_sql("select * from stock_list", con, index_col=['code'])
    stockList=delect_stop(stockList,begin_date)
    return stockList
peroid = 'D'
start_date = '2020-01-01'
end_date = '2024-01-01'

con1 = sqlite3.connect('etf.db')

dateList = get_period_date(peroid,start_date, end_date,con1)
print(len(dateList))
train_data=pd.DataFrame()
train_length = 600
sum_arrays = []
L=[]
for num in tqdm(range(train_length)):
    date = dateList[num+60]
    _start_date=dateList[num]
    stockList=get_stock(_start_date,con1)
    for i in stockList:
        df = get_price(i, _start_date, date, con1)
        df = df.dropna()
        if len(df) == 60:
            f_date=dateList[num+60+20]
            l = get_price(i, dateList[num+60],f_date,con1)
            l = l['low']
            change_percentage = ((np.mean(l.values) - df['close'][-1]) / df['close'][-1]) * 100
            if not np.isnan(change_percentage):
                L.append(change_percentage)
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(df)
                normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
                sum_arrays.append(normalized_df.values)
DATA = np.stack(sum_arrays, axis=0)
LABEL = np.stack(L)
print(DATA.shape)
print(LABEL.shape)
np.save('train_DATA.npy', DATA)
np.save('train_LABEL.npy', LABEL)