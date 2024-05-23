import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import tushare as ts
import time
from scipy.stats import linregress, pearsonr, spearmanr
pro = ts.pro_api('129dd5438782d33a4e811764b69ed0dbc9fc0c4e53fa4ee4d3718f7c')

# 指定中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义函数
def factor_effectiveness_analysis(date, stock_code, factor):
    regress_coeff = []
    t_value = []
    ics = []
    for i in range(len(date)-1):
        ret = []
        # 获取每只股票周期截面的收益率
        for code in stock_code:
            df = pro.daily(ts_code=code, end_date=date[i+1], limit='10',fields='close')
            ret.append(df.close[len(df)-1]/df.close[0]-1)

        # 计算因子与收益率的相关性、线性回归系数、t统计量
        correlation, slope, t_statistic =calculate_statistics_of_two_data(factor[i], ret)
        ics.append(correlation)
        t_value.append(t_statistic)
        regress_coeff.append(slope)

    # 计算平均IC值、IR值、负相关IC值占比、回归t检验均值
    average_ics = np.mean(ics) 
    ir = np.mean(ics)/np.std(ics) 
    negative_ratio = len(list(filter(lambda x: x < 0, ics)))/len(ics) 
    regress_t_mean = np.mean(t_value)

    IC = pd.DataFrame({'IC均值': [average_ics], 'ICIR值': [ir], '负相关IC值占比': [negative_ratio], '回归t检验均值': [regress_t_mean]})

    # 绘制IC值
    plt.figure(figsize=(10, 6))
    plt.plot(date[1:], ics, marker='o', linestyle='-')
    plt.title('IC值随日期变化折线图')
    plt.xlabel('日期')
    plt.ylabel('IC值')
    plt.ylim(-1, 1)
    plt.xticks(rotation=90)
    plt.savefig('IC-Date.png', dpi=300)
    plt.show()
    # 暂停以确保图表被渲染
    plt.pause(0.1)
    plt.close()

    # 绘制因子收益率图表
    plt.figure(figsize=(10, 6))
    plt.plot(date[1:], regress_coeff, label='factor_return', color='green')
    plt.xlabel('date')
    plt.ylabel('（cum_sum）return')
    plt.title('（累计）因子收益率随日期变化折线图')

    # 绘制因子收益率累积图表
    cumulative_sum = np.cumsum(regress_coeff)
    plt.plot(date[1:], cumulative_sum, label='cumulative_sum', color='blue')
    plt.legend(['因子收益率', '累计因子收益率'])
    plt.savefig('（累计）因子收益率随日期变化折线图.png', dpi=300)
    plt.show()

    return plt, IC


def factor_effectiveness_analysis2(date, stock_code, factor):
    p_values = []
    ics = []
    for i in range(len(date)-1):
        ret = []
        # 获取每只股票周期截面的收益率
        for code in stock_code:
            df = pro.daily(ts_code=code, end_date=date[i+1], limit='10',fields='close')
            ret.append(df.close[len(df)-1]/df.close[0]-1)

        # 计算因子与收益率的相关性、p值
        correlation, p_value = spearmanr(factor[i], ret)
        if np.isnan(correlation):
            correlation = 0
        ics.append(correlation)
        p_values.append(p_value)

    # 计算平均IC值、IR值、负相关IC值占比、p值
    average_ics = np.mean(ics) 
    ir = np.mean(ics)/np.std(ics) 
    negative_ratio = len(list(filter(lambda x: x < 0, ics)))/len(ics) 

    IC = pd.DataFrame({'IC均值': [average_ics], 'ICIR值': [ir], '负相关IC值占比': [negative_ratio], 'p值': [np.mean(p_values)]})

    # 绘制IC值
    plt.figure(figsize=(10, 6))
    plt.plot(date[1:], ics, marker='o', linestyle='-')
    plt.title('IC值随日期变化折线图')
    plt.xlabel('日期')
    plt.ylabel('IC值')
    plt.ylim(-1, 1)
    plt.xticks(rotation=90)
    plt.savefig('IC-Date.png', dpi=300)
    plt.show()
    # 暂停以确保图表被渲染
    plt.pause(0.1)
    plt.close()

    return plt, IC


def calculate_statistics_of_two_data(data1, data2):
    # 计算相关系数
    correlation, _ = pearsonr(data1, data2)

    # 线性回归，计算回归的t统计量和相应的p值
    slope, intercept, r_value, p_value, std_err= linregress(data1, data2)
    t_statistic = slope / std_err
    p_value_t = 2 * (1 - st.t.cdf(abs(t_statistic), len(data1) - 2))
    
    return  correlation, slope, t_statistic


# data1是排名性的值，data2是收益率序列
def rank_data_statistics(data1, data2):
    correlation, p_value = spearmanr(data1, data2)

    return correlation, p_value


'''
以下为实际运行结果
'''
data = pro.daily(ts_code='000001.SZ', start_date='20161231', end_date='20181231', fields='trade_date')

# 从 *本地数据* 获取函数所需的日期，股票代码和因子值
# start_date='20161231', end_date='20181231'

# data = pd.read_csv('/Users/gwo/Desktop/data/date_stock.csv')
# with open('data.pkl', 'wb') as f:
#     pickle.dump(data, f)
# try:
#     with open('data.pkl', 'rb') as f:
#         data = pickle.load(f)
# except FileNotFoundError:
#     # 如果文件不存在，则重新加载数据
#     data = pd.read_csv(f'/Users/gwo/Desktop/data/date_stock.csv')
#     with open('data.pkl', 'wb') as f:
#         pickle.dump(data, f)

# df1 = data[data.iloc[:,0].notnull()]
# df1.loc[:,'Date'] = pd.to_datetime(df1['Date']).dt.date.astype(str)

date = [] 
for i in range(0, len(data.trade_date), 10):
    date.append(data.trade_date[i])
date = date[::-1]

# stock_code = random.sample(list(df1['SecuCode']),10)
stock_code = ['000001.SZ', '600000.SH', '001872.SZ', '300458.SZ', '600200.SH']


# alpha1  0.136504  0.25189     0.375  0.545724
def alpha1(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')    
            factor[index].append(-np.corrcoef(df.vol, df.amount/df.vol)[0,1])
    factor = np.array(factor).T
    return factor

# alpha2  I -0.000968 -0.001659       0.5 -0.015842
def alpha2(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(np.mean(df.open/df.pre_close))
    factor = np.array(factor).T
    return factor 


# alpha3    0.027099  0.051316       0.5  0.115322
def alpha3(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(-df.vol[0]/np.mean(df.vol))
    factor = np.array(factor).T
    return factor


# alpha4    0.088241  0.167169  0.416667  0.182217
def alpha4(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(-np.corrcoef(df.high/df.low, df.amount/df.vol)[0,1])
    factor = np.array(factor).T
    return factor 


# alpha5    I -0.023153 -0.048838    0.5625  0.05498
def alpha5(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            df['x'] = ((df.close-df.low)-(df.high-df.close))/(df.high-df.low)
            factor[index].append(np.mean(df.x.diff()))
    factor = np.array(factor).T
    return factor 


# alpha6     I 0.059733  0.115882  0.458333  0.179574
def alpha6(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(np.mean((df.high[0:10]*df.low[0:10])**0.5-df.amount/df.vol))
    factor = np.array(factor).T
    return factor 


# alpha7    0.054904  0.091115  0.458333  0.33027
def alpha7(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='5',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(df.close[0]-df.close[4])
    factor = np.array(factor).T
    return factor 


# alpha8    0.032366  0.055938    0.4375  0.054289
def alpha8(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='12',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(np.mean(df.close[0:12])/df.close[0])
    factor = np.array(factor).T
    return factor 


# alpha9    -0.233333 -0.539761  0.541667 NAN
def alpha9(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        x4 = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='20',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            x1 = []
            x2 = []
            x3 = []
            for j in range(0,7):
                x1 = df.vol[j:j+10].rank(method='dense', ascending=False)
                x2 = (df.amount/df.vol)[j:j+10].rank(method='dense', ascending=False)
                x3.append(np.corrcoef(x1, x2)[0,1])
            x4 = pd.DataFrame([sum(x3[0:6]),sum(x3[1:7])]).rank(method='dense', ascending=False)
            factor[index].append(-x4.iloc[0,0])
    factor = np.array(factor).T
    return factor


# alpha10    0.019767  0.038946       0.5 -0.05336
def alpha10(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        x = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            x= df.vol[0:5].rank(method='dense', ascending=False)
            factor[index].append(np.corrcoef(df.high[0:5], x)[0,1])
    factor = np.array(factor).T
    return factor 


# alpha11   0.027663  0.050868       0.5  0.192946
def alpha11(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='24',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append((df.close[0]/np.mean(df.close[0:24]-1))*100)
    factor = np.array(factor).T
    return factor 


# alpha12   -0.074859 -0.154787  0.479167  0.523674
def alpha12(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        x4 = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='20',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            x1 = []
            x2 = []
            x3 = []
            for j in range(0,5):
                x1 = df.high[j:j+10].rank(method='dense', ascending=False)
                x2 = df.vol[j:j+10].rank(method='dense', ascending=False)
                x3.append(np.cov(x1, x2)[0,1])
            x4 = pd.DataFrame(x3).rank(method='dense', ascending=False)
            factor[index].append(-x4.iloc[0,0])
    factor = np.array(factor).T
    return factor


# alpha13   -0.027246 -0.058643  0.520833  0.041837
def alpha13(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='20',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(np.std(df.amount[0:20]))
    factor = np.array(factor).T
    return factor 


# alpha14   0.039386  0.072475  0.416667  0.152664
def alpha14(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        x1 = []
        x2 = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='12',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            x1 = df.open[0:10].rank(method='dense', ascending=False)
            x2 = df.vol[0:10].rank(method='dense', ascending=False)
            factor[index].append(-np.corrcoef(x1, x2)[0,1])
    factor = np.array(factor).T
    return factor 


# alpha15   0.060215  0.117619    0.4375  0.191924
def alpha15(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='12',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append((df.close[0]+df.high[0]+df.low[0])/3)
    factor = np.array(factor).T
    return factor 


# alpha16    0.074945  0.147811  0.458333  0.32919
def alpha16(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='12',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append((df.close[0]/df.close[11]-1)*df.vol[0])
    factor = np.array(factor).T
    return factor 


# alpha17      0.038443  0.074724    0.4375  0.106224
def alpha17(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(-np.corrcoef(df.open, df.vol)[0,1])
    factor = np.array(factor).T
    return factor 


# alpha18   -0.011122 -0.02164    0.4375  0.076953
def alpha18(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='30',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append((np.mean(df.vol[0:9])-np.mean(df.vol[0:26]))/np.mean(df.vol[0:12])*100)
    factor = np.array(factor).T
    return factor 


# alpha19   0.012796  0.024832       0.5  0.021523
def alpha19(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='10',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            factor[index].append(df.pct_chg[0]*df.vol[0])
    factor = np.array(factor).T
    return factor 


# alpha20   -0.048079 -0.092259  0.479167 -0.098371
def alpha20(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='30',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            x = []
            for j in range(0,5):
                x.append(np.mean(df.vol[j:20+j]))
            factor[index].append(np.corrcoef(x, df.low[0:5])[0,1]+(df.high[0]+df.low[0])/2-df.close[0])
    factor = np.array(factor).T
    return factor 


# alpha21 -0.058184 -0.101134  0.604167 -0.162635
def alpha21(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='160',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            df['x'] = df.high/df.low-1
            factor[index].append(sum(df.nsmallest(112, 'x').pct_chg))
    factor = np.array(factor).T
    return factor 


# alpha22   -0.09062 -0.160745     0.625 -0.225411
def alpha22(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='20',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            df['x'] = df.amount/df.vol
            factor[index].append(sum(df.nlargest(10, 'x').pct_chg)-sum(df.nsmallest(10, 'x').pct_chg))
    factor = np.array(factor).T
    return factor 


# alpha23   0.017605  0.031133  0.458333  0.094181
def alpha23(date, stock_code):
    factor = [[] for _ in range(len(stock_code))]
    for index, code in enumerate(stock_code):
        factor[index] = []
        for i in range(len(date)-1):
            df = pro.daily(ts_code=code, end_date=date[i], limit='20',fields='open,close,high,low,pre_close,change,pct_chg,vol,amount')
            df['x'] = df.high/df.low-1
            factor[index].append(np.mean(df.nlargest(5, 'close').x)-np.mean(df.nsmallest(5, 'close').x))
    factor = np.array(factor).T
    return factor 


# 因子合成为Alpha   -0.012253 -0.024325  0.479167 -0.024915；  0.033224  0.073045  0.479167  0.114775
def Alpha(date, stock_code):
    factor10 = alpha10(date, stock_code)
    print('alpha10 done')
    time.sleep(10)

    factor23 = alpha23(date, stock_code)
    print('alpha23 done')
    time.sleep(10)

    factor1 = alpha1(date, stock_code)
    print('alpha1 done')
    time.sleep(10)

    factor2 = alpha2(date, stock_code)
    print('alpha2 done')
    time.sleep(10)

    factor3 = alpha3(date, stock_code)
    print('alpha3 done')
    time.sleep(10)

    factor4 = alpha4(date, stock_code)
    print('alpha4 done')
    time.sleep(10)

    factor5 = alpha5(date, stock_code)
    print('alpha5 done')
    time.sleep(10)

    factor6 = alpha6(date, stock_code)
    print('alpha6 done')
    time.sleep(10)

    factor7 = alpha7(date, stock_code)
    print('alpha7 done')
    time.sleep(10)

    factor8 = alpha8(date, stock_code)
    print('alpha8 done')
    time.sleep(10)

    factor9 = alpha9(date, stock_code)
    print('alpha9 done')
    time.sleep(10)

    factor11 = alpha11(date, stock_code)
    print('alpha11 done')
    time.sleep(10)

    factor12 = alpha12(date, stock_code)
    print('alpha12 done')
    time.sleep(10)

    factor13 = alpha13(date, stock_code)
    print('alpha13 done')
    time.sleep(10)

    factor14 = alpha14(date, stock_code)
    print('alpha14 done')
    time.sleep(10)

    factor15 = alpha15(date, stock_code)
    print('alpha15 done')
    time.sleep(10)

    factor16 = alpha16(date, stock_code)
    print('alpha16 done')
    time.sleep(10)

    factor17 = alpha17(date, stock_code)
    print('alpha17 done')
    time.sleep(10)

    factor18 = alpha18(date, stock_code)
    print('alpha18 done')
    time.sleep(10)

    factor19 = alpha19(date, stock_code)
    print('alpha19 done')
    time.sleep(10)

    factor20 = alpha20(date, stock_code)
    print('alpha20 done')
    time.sleep(10)

    factor21 = alpha21(date, stock_code)
    print('alpha21 done')
    time.sleep(10)

    factor22 = alpha22(date, stock_code)
    print('alpha22 done')
    time.sleep(10)


    # 合并因子
    factor = [factor1,factor2,factor3,factor4,factor5,factor6,factor7,factor8,factor9,factor10,factor11,factor12,factor13,factor14,factor15,factor16,factor17,factor18,factor19,factor20,factor21,factor22,factor23]
    factor = np.array(factor)

    # 以IC值作为权重，计算加权因子
    ic = [0.136504,-0.000968,0.027099,0.088241,-0.023153,0.059733,0.054904,0.032366,-0.233333,0.019767,0.027663,-0.074859,-0.027246,
        0.039386,0.060215,0.074945,0.038443,-0.011122,0.012796,-0.048079,-0.058184,-0.09062,0.017605]
    ic = np.array(ic)
    ic[abs(ic)<0.02] = 0
    ic = ic / np.sum(ic)
    weighted_factors = factor * ic[:, np.newaxis, np.newaxis]
    Alpha = np.sum(weighted_factors, axis=0)

    return Alpha

factor = Alpha(date, stock_code)
# 调用函数
plt, IC = factor_effectiveness_analysis(date, stock_code, factor)
plt.show()
print(IC)
