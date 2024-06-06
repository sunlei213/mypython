# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:25:23 2024

@author: sunlei
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
import torch.nn as nn
from tqdm import tqdm
import sqlite3
from pandas.io import sql
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

class model(nn.Module):
    def __init__(self,
                 fc1_size=2000,
                 fc2_size=1000,
                 fc3_size=100,
                 fc1_dropout=0.2,
                 fc2_dropout=0.2,
                 fc3_dropout=0.2,
                 num_of_classes=50):
        super(model, self).__init__()

        self.f_model = nn.Sequential(
            nn.Linear(5088, fc1_size),  # 887
            nn.BatchNorm1d(fc1_size),
            nn.ReLU(),
            nn.Dropout(fc1_dropout),
            nn.Linear(fc1_size, fc2_size),
            nn.BatchNorm1d(fc2_size),
            nn.ReLU(),
            nn.Dropout(fc2_dropout),
            nn.Linear(fc2_size, fc3_size),
            nn.BatchNorm1d(fc3_size),
            nn.ReLU(),
            nn.Dropout(fc3_dropout),
            nn.Linear(fc3_size, 1),

        )

        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )

        self.conv_2D = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout(fc3_dropout),
            nn.ReLU(),
        )
        hidden_dim = 32
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                            # dropout=fc3_dropout,
                            bidirectional=True)
        hidden_dim = 1
        self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                         # dropout=fc3_dropout,
                         bidirectional=True)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        apply = torch.narrow(x, dim=-1, start=0, length=1).squeeze(1)
        redeem = torch.narrow(x, dim=-1, start=1, length=1).squeeze(1)
        apply, _ = self.l(apply)
        redeem, _ = self.l(redeem)
        apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
        redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))

        ZFF = torch.narrow(x, dim=-1, start=2, length=1).squeeze(1)
        HS = torch.narrow(x, dim=-1, start=3, length=1).squeeze(1)
        ZFF, _ = self.l(ZFF)
        HS, _ = self.l(HS)
        ZFF = torch.reshape(ZFF, (ZFF.shape[0], ZFF.shape[1] * ZFF.shape[2]))
        HS = torch.reshape(HS, (HS.shape[0], HS.shape[1] * HS.shape[2]))
        
        xx = x.unsqueeze(1)
        xx = self.conv_2D(xx)
        xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))
        x = x.transpose(1, 2)
        x = self.conv_layers1(x)
        out = x.transpose(1, 2)
        out2, _ = self.lstm(out)
        out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

        IN = torch.cat((xx, out2, apply, redeem, ZFF, HS), dim=1)
        out = self.f_model(IN)
        return out

def get_price(stock, start, end, con):
    sql_st = "select distinct * from '{0}' where date between '{1}' and '{2}' order by date".format(
        stock, start, end)
    df = sql.read_sql(sql_st, con, index_col=['date'])
    return df

#获取指定周期的日期列表 'W、M、Q'
def get_period_date(peroid,start_date, end_date, con):
    #设定转换周期period_type  转换为周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'
    stock_data = get_price('510050.XSHG',start_date,end_date,con)
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.index = stock_data.index.to_pydatetime()
    #记录每个周期中最后一个交易日
    #stock_data['date']=stock_data.index
    #进行转换，周线的每个变量都等于那一周中最后一个交易日的变量值
    period_stock_data=stock_data.resample(peroid).last()
    period_stock_data = period_stock_data.dropna()
    #stock_data.index = pd.to_datetime(stock_data.index)
    date=period_stock_data.index
    pydate_array = date.to_pydatetime()
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)

    date_only_series = pd.Series(date_only_array)

    #start_date = datetime.strptime(start_date, "%Y-%m-%d")
    #start_date=start_date-timedelta(days=1)
    #start_date = start_date.strftime("%Y-%m-%d")
    date_list=date_only_series.values.tolist()
    #date_list.insert(0,start_date)
    return date_list
#去除上市距beginDate不足3个月的ETF
def delect_stop(stocklist,beginDate,n=30*3):
    stocks = stocklist.index.tolist()
    stockList=[]
    beginDate = datetime.strptime(beginDate, "%Y-%m-%d")
    for stock in stocks:
        if stock[0:3] == '511': continue
        start_date=stocklist.loc[stock,'start_date']
        if start_date<(beginDate-timedelta(days=n)).strftime('%Y-%m-%d'):
            stockList.append(stock)
    return stockList

def get_stock(begin_date,con):
    stockList = sql.read_sql("select * from stock_list", con, index_col=['code'])
    stockList=delect_stop(stockList,begin_date)
    return stockList

def choice_stock(model, cu_date, con):
    sum_arrays = []
    L=[]
    dateList = get_period_date('D','2022-01-01', cu_date,con)
    date = dateList[-1]
    _start_date=dateList[-61]
    
    stockList=get_stock(date,con)
    for i in stockList:
        df = get_price(i, _start_date, date, con)
        df = df.dropna()
        #print(len(df))
        if len(df) == 60:
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(df)
            normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
            sum_arrays.append(normalized_df.values)
            L.append(i)
    DATA = np.stack(sum_arrays, axis=0)
    DATA = torch.tensor(DATA, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(DATA)
    numpy_array = output.numpy()
    
    # 创建 DataFrame
    df = pd.DataFrame({'name': L, '打分': numpy_array.flatten()})
    # 获取打分最高的 10 个 name
    top_10_names = df.nlargest(10, '打分')
    return top_10_names

peroid = '3D'
start_date = '2024-01-01'
end_date = '2024-06-01'

con1 = sqlite3.connect('etf.db')
state_dict = torch.load("model_baseline.pt", map_location=torch.device("cpu"))
model = model()  
model.load_state_dict(state_dict)
stocks={}
dateList = get_period_date(peroid,start_date, end_date,con1)
print(len(dateList))
for date in tqdm(dateList):
    sl = choice_stock(model, date, con1)
    stocks[date] = list(sl['name'])
print(stocks)