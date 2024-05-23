import csv
from collections import defaultdict
from datetime import datetime
import pickle

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
data = load_data(f'/Users/gwo/Desktop/data/日行情/')
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# 获取某一天所有股票的数据
def get_prices_for_day(date, index):
    date = datetime.strptime(date, '%Y-%m-%d')  # 修改日期格式为'YYYY-MM-DD'
    prices = {stock_code: info for stock_code, info in index[date].items()}
    return prices

