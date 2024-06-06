# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:40:13 2017

@author: sunlei
"""

import sqlite3
from pandas.io import sql
from datetime import datetime, timedelta
import tushare as ts
import traceback

con1 = sqlite3.connect('stocks.db')
today = datetime.now().strftime('%Y-%m-%d')
cur = con1.cursor()
get_ls_ts = False
get_st = True
try:
    if get_ls_ts:
        df = ts.get_stock_basics()
    else:
        df = sql.read_sql("select * from stock_list", con1, index_col=['code'])
    stock_list = df.index
except:
    traceback.print_exc()
    get_st = False
if get_st:
    if get_ls_ts:
        sql.to_sql(df, name='stock_list', con=con1, if_exists='replace')
    len1 = len(stock_list)
    len1 = int(len1 / 100)
    print(len1)
    i1 = 0
    j1 = 0
    for ticker in stock_list:
        i1 += 1
        if (i1 % len1) == 0:
            j1 += 1
            print('\r数据生成{0}%'.format(j1), end='')
        if sql.has_table(ticker, con1):
            sql_st = "select date from '{0}' order by date desc".format(ticker)
            cur.execute(sql_st)
            rec1 = cur.fetchone()
            endday = rec1[0]
            endday = (datetime.strptime(endday, "%Y-%m-%d") +
                      timedelta(days=1)).strftime("%Y-%m-%d")
            if endday > today:
                continue
            r_or_a = 'append'
        else:
            endday = '2013-01-01'
            r_or_a = 'replace'
        try:
            df = ts.get_hist_data(ticker, start=endday, end=today)
        except:
            continue
        if str(type(df)) != "<class 'NoneType'>":
            df = df.dropna().sort_index()
            sql.to_sql(df, name=ticker, con=con1, if_exists=r_or_a)
    zs_list = ['sh', 'sz', 'hs300', 'sz50', 'zxb', 'cyb']
    for ticker in zs_list:
        if sql.has_table(ticker, con1):
            sql_st = "select date from '{0}' order by date desc".format(ticker)
            cur.execute(sql_st)
            rec1 = cur.fetchone()
            endday = rec1[0]
            endday = (datetime.strptime(endday, "%Y-%m-%d") +
                      timedelta(days=1)).strftime("%Y-%m-%d")
            if endday > today:
                continue
            r_or_a = 'append'
        else:
            endday = '2013-01-01'
            r_or_a = 'replace'
        try:
            df = ts.get_hist_data(ticker, start=endday, end=today)
        except:
            continue
        if str(type(df)) != "<class 'NoneType'>":
            df = df.dropna().sort_index()
            sql.to_sql(df, name=ticker, con=con1, if_exists=r_or_a)

con1.commit()
con1.close()
print('数据添加完毕')
