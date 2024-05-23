import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import defaultdict
from datetime import datetime
import pickle
import csv

# 指定中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(directory):
    data = defaultdict(dict)
    index = defaultdict(dict)
    for year in range(2017, 2019):  # 从2017年到2018年
        print(f'load daily quote data in {year}')
        with open(f'{directory}/daily_quote_{year}.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                date = datetime.strptime(row['TradingDay'], '%Y-%m-%d')  # 修改日期格式为'YYYY-MM-DD'
                date = date.strftime('%Y-%m-%d')
                stock_code = row['SecuCode']
                close_price_str = row['ClosePrice']
                open_price_str = row['OpenPrice']
                high_price_str = row['HighPrice']
                low_price_str = row['LowPrice']
                prev_close_price_str = row['PrevClosePrice']
                trading_volumes_str = row['TradingVolumes']
                turnover_value_str = row['TurnoverValue']
                ret_str = row['ret']
                non_restricted_shares_str = row['NonRestrictedShares']
                a_floats_str = row['AFloats']
                zz500_ret_str = row['ZZ500Ret']
                ashares_str = row['Ashares']
                pt_flag_str = row['PTFlag']
                # 检查是否为空字符串，如果是则跳过该行
                if any(val == '' for val in
                       [close_price_str, open_price_str, high_price_str, low_price_str, prev_close_price_str,
                        trading_volumes_str, turnover_value_str, ret_str, non_restricted_shares_str, a_floats_str,
                        zz500_ret_str, ashares_str, pt_flag_str]):
                    continue
                close_price = float(close_price_str)
                open_price = float(open_price_str)
                high_price = float(high_price_str)
                low_price = float(low_price_str)
                prev_close_price = float(prev_close_price_str)
                trading_volumes = int(float(trading_volumes_str))
                if trading_volumes == 0:
                    continue
                turnover_value = float(turnover_value_str)
                ret = float(ret_str)
                non_restricted_shares = int(float(non_restricted_shares_str))
                a_floats = float(a_floats_str)
                zz500_ret = float(zz500_ret_str)
                ashares = float(ashares_str)
                pt_flag = int(float(pt_flag_str))
                key = (date, stock_code)
                data[key] = {
                    'open': open_price,
                    'close': close_price,
                    'high': high_price,
                    'low': low_price,
                    'prev_close': prev_close_price,
                    'volume': trading_volumes,
                    'turnover': turnover_value,
                    'return': ret,
                    'non_restricted_shares': non_restricted_shares,
                    'a_floats': a_floats,
                    'zz500_ret': zz500_ret,
                    'ashares': ashares,
                    'pt_flag': pt_flag
                }
                index[date][stock_code] = {
                    'open': open_price,
                    'close': close_price,
                    'high': high_price,
                    'low': low_price,
                    'prev_close': prev_close_price,
                    'volume': trading_volumes,
                    'turnover': turnover_value,
                    'return': ret,
                    'non_restricted_shares': non_restricted_shares,
                    'a_floats': a_floats,
                    'zz500_ret': zz500_ret,
                    'ashares': ashares,
                    'pt_flag': pt_flag
                }
    return data, index

# 加载数据并持久化
#data, index = load_data(f'/Users/gwo/Desktop/data/日行情/')
#with open('data.pkl', 'wb') as f:
#    pickle.dump((data, index), f)
try:
    with open('data.pkl', 'rb') as f:
        data, index = pickle.load(f)
except FileNotFoundError:
    # 如果文件不存在，则重新加载数据和索引
    data, index = load_data(f'/Users/gwo/Desktop/data/日行情/')
    with open('data.pkl', 'wb') as f:
        pickle.dump((data, index), f)
        
# 获取某一天所有股票的数据
def get_prices_for_day(date, index):
    date = datetime.strptime(date, '%Y-%m-%d')  # 修改日期格式为'YYYY-MM-DD'
    date = date.strftime('%Y-%m-%d')
    print(date)
    prices = {stock_code: info for stock_code, info in index[date].items()}
    return prices


prices = (get_prices_for_day('2017-12-29',index))
print(prices['000001']['volume'])
# 定义函数
def factor_effectiveness_analysis(date, stock_code, factor):
    ics = []
    # for i in range(1,len(date)):
    #     ret = []
    #     # 获取每只股票周期截面的收益率
    #     for code in stock_code:
    #         stock_data = index[date[i]].get(code)
    #         if stock_data:
    #             close1 = stock_data['close']
    #             close0 = get_prices_for_day(date[i-1], index)[code]['close']
    #         ret.append(close1/close0-1)

    
    for i in range(2, len(date)):
        ret = []
    # 获取每只股票周期截面的收益率
        for code in stock_code:
            stock_data = index[date[i]].get(code)
            prev_day_prices = index[date[i-1]].get(code)
            if stock_data and prev_day_prices:
                close1 = stock_data['close']
                close0 = prev_day_prices['close']
                ret.append(close1/close0-1)

        # 计算因子与收益率的相关性
        if len(factor[i-1])==len(ret):    
            ic=st.pearsonr(factor[i-1],ret)
            ics.append(ic[0])

    # 计算平均IC值、IR值、负相关IC值占比
    average_ics = np.mean(ics) 
    ir = np.mean(ics)/np.std(ics) 
    #negative_ratio = len(list(filter(lambda x: x < 0, ics)))/len(ics) 
    #IC = pd.DataFrame({'IC均值': [average_ics], 'IR值': [ir], '负相关IC值占比': [negative_ratio]})

    # 使用Matplotlib创建折线图
    plt.figure(figsize=(10, 6))
    plt.plot(date, ics, marker='o', linestyle='-')
    plt.title('IC值随日期变化折线图')
    plt.xlabel('日期')
    plt.ylabel('IC值')
    plt.ylim(-1, 1)
    plt.xticks(rotation=90)

    return plt, IC

    
# 获取函数所需的日期，股票代码和因子值
df = pd.read_csv('/Users/gwo/Desktop/data/date_stock.csv')
df1 = df[df.iloc[:,0].notnull()]
df1['Date'] = pd.to_datetime(df1['Date']).dt.date.astype(str)

date = [] 
# 每隔d天取一个日期作为交易日
for i in range(0, len(df1), 30):
        date.append(df1['Date'][i])

stock_code = df1['SecuCode'][0:2]


factor = [[] for _ in range(len(stock_code))]
for x, code in enumerate(stock_code):
    factor[x] = []
    volume = []
    turnover = []
    for i in range(1,len(date)):
        for day in df1.Date[df1.Date.index[30*(i-1)]:df1.Date.index[30*i]]:
            stock_data = index[day].get(code)
            if stock_data:
                volume.append(get_prices_for_day(day,index)[code]['volume']) 
                turnover.append(get_prices_for_day(day,index)[code]['turnover'])
            else:
                continue
        turnover_volume_ratio = [turnover[i]/volume[i] for i in range(len(volume))]
        corr_coef = -np.corrcoef(volume, turnover_volume_ratio)[0, 1]
        factor[x].append(corr_coef)
factor = np.array(factor).T


# 调用函数
plt, IC = factor_effectiveness_analysis(date, stock_code, factor)
plt.show()
print(IC)

