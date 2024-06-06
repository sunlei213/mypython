# 导入函数库
from jqdata import *
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:52:48 2017

@author: sunlei
"""

import numpy as np
from keras.models import load_model
import os
import pandas as pd
import json
import talib
from functools import reduce
from datetime import datetime, timedelta
import sqlite3
from pandas.io import sql

HISTORY = 20
today = datetime.now()
endday = today.strftime('%Y-%m-%d')
startday = (today + timedelta(days=-120)).strftime('%Y-%m-%d')
print(startday, endday)


def normal(data, max1):
    return data / max1


def SMA_CN(close, timeperiod):
    close = np.nan_to_num(close)
    return reduce(lambda x, y: ((timeperiod - 1) * x + y) / timeperiod, close)


# rsi
def RSI_CN(close, timeperiod=10):
    diff = list(map(lambda x, y: x - y, close[1:], close[:-1]))
    diffGt0 = list(map(lambda x: 0 if x < 0 else x, diff))
    diffABS = list(map(lambda x: abs(x), diff))
    diff = np.array(diff)
    diffGt0 = np.array(diffGt0)
    diffABS = np.array(diffABS)
    diff = np.append(diff[0], diff)
    diffGt0 = np.append(diffGt0[0], diffGt0)
    diffABS = np.append(diffABS[0], diffABS)
    rsi = np.nan_to_num(list(map(lambda x: SMA_CN(diffGt0[:x], timeperiod) / SMA_CN(diffABS[:x], timeperiod) * 100, range(1, len(diffGt0) + 1))))

    return np.array(rsi)


# KDJ
def KDJ(security_data, fastk_period=5, slowk_period=3, fastd_period=3):

    # 计算 KDJ

    high = np.array(security_data['high'])
    low = np.array(security_data['low'])
    close = np.array(security_data['close'])
    kValue, dValue = talib.STOCHF(
        high, low, close, fastk_period, fastd_period, fastd_matype=0)
    kValue = np.array(
        list(
            map(lambda x: SMA_CN(kValue[:x], slowk_period),
                range(1,
                      len(kValue) + 1))))
    dValue = np.array(
        list(
            map(lambda x: SMA_CN(kValue[:x], fastd_period),
                range(1,
                      len(kValue) + 1))))
    jValue = 3 * kValue - 2 * dValue

    func = lambda arr: np.array([0 if x < 0 else (100 if x > 100 else x) for x in arr])

    k = func(kValue)
    d = func(dValue)
    j = func(jValue)
    return k, d, j


# MACD
def MACD(security_data, fastperiod=12, slowperiod=26, signalperiod=9):

    # 计算 MACD

    macd_DIF, macd_DEA, macd = talib.MACDEXT(
        np.array(security_data['close']),
        fastperiod=fastperiod,
        fastmatype=1,
        slowperiod=slowperiod,
        slowmatype=1,
        signalperiod=signalperiod,
        signalmatype=1)
    macd_HIST = macd
    return macd_DIF, macd_DEA, macd_HIST


# BIAS
def BIAS(security_data):

    # 计算 BIAS
    average_price = np.array(security_data['ma5'])
    current_price = np.array(security_data['close'])
    bias = (current_price - average_price) / average_price
    return bias


def get_data1(stock, start, end, con):
    sql_st = "select distinct * from '{0}' where date between '{1}' and '{2}' order by date".format(
        stock, start, end)
    df = sql.read_sql(sql_st, con, index_col=['date'])
    return df


def make_data(universe, con):
    len1 = len(universe)
    len1 = int(len1 / 100)
    print(len1)
    i1 = 0
    j1 = 0
    st_list = []
    samples1 = []
    samples2 = []
    samples3 = []
    for ticker in universe:  # 遍历每支股票
        #df = ts.get_hist_data(ticker,start='2017-02-01',end=endday)
        i1 += 1
        if (i1 % len1) == 0:
            j1 += 1
            print('\r数据生成{0}%'.format(j1), end='')

        if not sql.has_table(ticker, con):
            print('\r{0} table is not exist'.format(ticker), end="")
            continue
        df = get_data1(ticker, start=startday, end=endday, con=con)
        if str(type(df)) != "<class 'NoneType'>":
            # raw_data = df.dropna().sort_index()
            df = df.dropna()
            #           if 'turnover' not in df.keys():
            #                print('\r正在生成股票{0}\n'.format(ticker), end='')
            #                continue
            if df.shape[0] < 60:
                continue
            k, d, j = KDJ(df)
            dif, dea, macd = MACD(df)
            ma20 = np.array(df['ma20'])
            ma5 = np.array(df['ma5'])
            close_pr = np.array(df['close'])
            cl5 = list(close_pr / ma5 - 1)
            cl20 = list(close_pr / ma20 - 1)
            ma5 = (ma5[1:] / ma5[:-1] - 1)
            ma5 = [0] + list(normal(ma5, 0.1))
            ma20 = (ma20[1:] / ma20[:-1] - 1)
            ma20 = [0] + list(normal(ma20, 0.1))
            bias = BIAS(df)
            rsi10 = RSI_CN(df['close'])
            rsi20 = RSI_CN(df['close'], 20)
            rsi10 = normal(rsi10, 100)
            rsi20 = normal(rsi20, 100)
            k = normal(k, 100)
            d = normal(d, 100)
            dif = normal(dif, 10)
            macd = normal(macd, 10)
            bias = normal(bias, 0.2)
            hsl = normal(df['turnover'], 100)
            rate = (close_pr[1:] / close_pr[:-1] - 1)
            rate = [0] + list(normal(rate, 0.11))
            nan_c = np.max([
                list(np.isnan(dif)).count(True),
                list(np.isnan(d)).count(True),
                list(np.isnan(hsl)).count(True)
            ])
            if len(list(k)) >= nan_c + HISTORY:
                k = k[-HISTORY:]
            else:
                print("k len:{0}".format(len(list(k))))
                continue
            if len(list(d)) >= nan_c + HISTORY:
                d = d[-HISTORY:]
            else:
                print("d len:{0}".format(len(list(d))))
                continue
            if len(list(dif)) >= nan_c + HISTORY:
                dif = dif[-HISTORY:]
            else:
                print("dif len:{0}".format(len(list(dif))))
                continue
            if len(list(macd)) >= nan_c + HISTORY:
                macd = macd[-HISTORY:]
            else:
                print("macd len:{0}".format(len(list(macd))))
                continue
            if len(list(bias)) >= nan_c + HISTORY:
                bias = bias[-HISTORY:]
            else:
                print("bias len:{0}".format(len(list(bias))))
                continue
            if len(list(rsi10)) >= nan_c + HISTORY:
                rsi10 = rsi10[-HISTORY:]
            else:
                print("rsi10 len:{0}".format(len(list(rsi10))))
                continue
            if len(list(rsi20)) >= nan_c + HISTORY:
                rsi20 = rsi20[-HISTORY:]
            else:
                print("bias len:{0}".format(len(list(rsi20))))
                continue
            if len(list(cl5)) >= nan_c + HISTORY:
                cl5 = cl5[-HISTORY:]
            else:
                print("rate cl5:{0}".format(len(list(cl5))))
                continue
            if len(list(cl20)) >= nan_c + HISTORY:
                cl20 = cl20[-HISTORY:]
            else:
                print("rate cl20:{0}".format(len(list(cl20))))
                continue
            if len(list(ma5)) >= nan_c + HISTORY:
                ma5 = ma5[-HISTORY:]
            else:
                print("rate ma5:{0}".format(len(list(ma5))))
                continue
            if len(list(ma20)) >= nan_c + HISTORY:
                ma20 = ma20[-HISTORY:]
            else:
                print("rate ma20:{0}".format(len(list(ma20))))
                continue
            if len(list(hsl)) >= nan_c + HISTORY:
                hsl = list(hsl[-HISTORY:])
            else:
                print("hsl len:{0}".format(len(list(hsl))))
                continue
            if len(list(rate)) >= nan_c + HISTORY:
                rate = list(rate[-HISTORY:])
            else:
                print("rate len:{0}".format(len(list(rate))))
                continue
            sample = np.zeros((HISTORY, 9))
            sample1 = np.zeros((HISTORY, 7))
            sample2 = np.zeros((HISTORY, 13))
            for num in range(HISTORY):
                macd1 = macd[num]
                dif1 = dif[num]
                cl_5 = cl5[num]
                cl_20 = cl20[num]
                if macd1 > 1 or macd1 < -1:
                    macd1 = macd1 / abs(macd1)
                if dif1 > 1 or dif1 < -1:
                    dif1 = dif1 / abs(macd1)
                if cl_5 > 1 or cl_5 < -1:
                    cl_5 = cl_5 / abs(cl_5)
                if cl_20 > 1 or cl_20 < -1:
                    cl_20 = cl_20 / abs(cl_20)
                sample[num] = np.array([
                    rate[num], hsl[num], bias[num], macd1, dif1, k[num],
                    d[num], rsi10[num], rsi20[num]
                ])
                sample1[num] = np.array([
                    rate[num], hsl[num], bias[num], macd1, dif1, k[num], d[num]
                ])
                sample2[num] = np.array([
                    rate[num], hsl[num], cl_5, cl_20, ma5[num], ma20[num],
                    bias[num], macd1, dif1, k[num], d[num], rsi10[num],
                    rsi20[num]
                ])
            samples1.append(sample)
            samples2.append(sample1)
            samples3.append(sample2)
            st_list.append(ticker)

    return np.array(samples1), np.array(samples2), np.array(
        samples3), np.array(st_list)


def predict_point_by_point(model, data):
    # 每次只预测1步长
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    return predicted


s1 = r'^300\d+'
# df = ts.get_stock_basics()
con1 = sqlite3.connect('stocks.db')
cur = con1.cursor()
df = sql.read_sql(
    "select * from stock_list where esp > 0", con1, index_col=['code'])

cyb = df[df.index.str.contains(s1)]

s1 = r'^N\w+|^\*\w+|^ST\w+'
st = df[df['name'].str.contains(s1)]
stocks = set(df.sort_values('totalAssets').index)
# univ = stocks-set(cyb.index)-set(st.index)
univ = stocks - set(st.index)
X_test1, X_test2, X_test3, Y_test = make_data(univ, con1)
X_test1 = X_test1.reshape(X_test1.shape[0], 20, 9)
X_test2 = X_test2.reshape(X_test2.shape[0], 20, 7)
X_test3 = X_test3.reshape(X_test3.shape[0], 20, 13)

mod_file = 'lstm9.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test1)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm9.txt', 'w'))

mod_file = 'lstm400.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test1)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm400.txt', 'w'))

mod_file = 'lstm9-128-4.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test1)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm7.txt', 'w'))

mod_file = 'lstm400_13.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test3)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm400_13.txt', 'w'))

mod_file = 'lstm13-128-4_1.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test3)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm13-128-4_1.txt', 'w'))

mod_file = 'lstm13-512-256-128-4.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test3)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm13-128-4.txt', 'w'))

mod_file = 'lstm13-128-4.h5'

if os.path.exists(mod_file):
    print('开始预测')
    model = load_model(mod_file)
else:
    print('lstm不存在')
    exit()
predictions = predict_point_by_point(model, X_test3)
pre = pd.DataFrame(predictions, index=Y_test, columns=['rate'])
final = pre.sort_values('rate', ascending=False)
str1 = final.index[:50]
sl1 = []
for i in range(len(str1)):
    st1 = str1[i]
    if st1[:2] in ('00', '30'):
        sl1.append(st1 + '.XSHE')
    else:
        sl1.append(st1 + '.XSHG')
print(final[:50], sl1)
json.dump(sl1, open('lstm13-512-256-128-4.txt', 'w'))
