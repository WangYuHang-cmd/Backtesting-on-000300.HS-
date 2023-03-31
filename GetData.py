import tushare as ts
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 获取沪深300成份股
pro = ts.pro_api(token='2cb7b30105056801542e9ff02a8370c509b84397686c94376801c170')
hs300 = pro.index_weight(index_code='000300.SH', start_date='20220301', end_date='20220331')

# 获取每只股票最近一个月的日线数据
df = pd.DataFrame()
for ts_code in hs300.con_code.tolist():
    data = pro.daily(ts_code=ts_code, start_date='20220301', end_date='20220331')
    # df = df.append(data)
    df = pd.concat([df, data], ignore_index=True)

# 转换日期格式，并按日期升序排序
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values(by='trade_date', ascending=True)

# 计算每股盈余和市值
df_basic = pro.daily_basic(ts_code=','.join(hs300.con_code.tolist()), trade_date='20220331',
                            fields='ts_code,trade_date,pe,pb,total_mv')


# 将时间列转换为 datetime64[ns] 类型
df['trade_date'] = pd.to_datetime(df['trade_date'])
df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])

# 合并数据，并且按照每股盈余和市值排序
df_merged = pd.merge(df, df_basic, on=['ts_code', 'trade_date'], how='left')
df_merged = df_merged.dropna(subset=['pe', 'pb', 'total_mv'])
df_merged['eps'] = df_merged['close'] / df_merged['pe']
df_merged['circ_mv'] = df_merged['total_mv'] / df_merged['pb']
df_merged = df_merged.sort_values(by=['eps', 'circ_mv'], ascending=False) #[False, True]

# 按照每股盈余和市值分组，并计算收益率
group_num = 10
group_size = len(df_merged) // group_num
df_merged['group'] = pd.cut(np.arange(len(df_merged)), bins=group_num, labels=False)
df_merged['group_return'] = df_merged.groupby(['group'])['pct_chg'].cumsum()

# 输出回测结果
group_return = df_merged.groupby(['group'])['group_return'].tail(1)
group_return.index = np.arange(1, group_num+1)
print(group_return)

# 绘制每组回测结果的可视化图表
plt.figure(figsize=(8, 6))
for i in range(group_num):
    group_return_i = df_merged[df_merged['group'] == i]['group_return']
    plt.plot(group_return_i, label='group {}'.format(i+1))
plt.legend()
plt.title('Group Return')
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Return')
plt.show()


# 计算每个组别的收益率
# grouped = df_merged.groupby(['group', 'trade_date'])
# cum_profit = grouped['pct_chg'].apply(lambda x: (x / 100 + 1).cumprod()[-1])
#
# # 将累计收益率保存到数据框中
# df_profit = cum_profit.reset_index()
# df_profit = df_profit.pivot(index='trade_date', columns='group', values='pct_chg')
# df_profit = df_profit.fillna(method='ffill')
#
# sns.lineplot(data=df_profit, x='trade_date', y='cumulative_profit', hue='group', palette='Set2')
# plt.title('Cumulative Profits by EPS and Market Value Group')
# plt.xlabel('Trade Date')
# plt.ylabel('Cumulative Profit')
# plt.show()