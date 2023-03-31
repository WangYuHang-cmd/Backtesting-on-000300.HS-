import tushare as ts
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 获取沪深300成份股
pro = ts.pro_api(token='2cb7b30105056801542e9ff02a8370c509b84397686c94376801c170')
hs300 = pro.index_weight(index_code='000300.SH', start_date='20220301', end_date='20220331')

# 获取每只股票最近一个月的日线数据
df = pd.DataFrame()
for ts_code in hs300.con_code.tolist():
    data = pro.daily(ts_code=ts_code, start_date='20220301', end_date='20220331')
    df = pd.concat([df, data], ignore_index=True)

# 转换日期格式，并按日期升序排序
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values(by='trade_date', ascending=True)

# 计算每股盈余和市值
df_basic = pro.daily_basic(ts_code=','.join(hs300.con_code.tolist()), trade_date='20220331',
                            fields='ts_code,trade_date,pe,pb,total_mv')

# 将时间列转换为 datetime64[ns] 类型
df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])

# 合并数据，并且按照每股盈余和市值排序
df_merged = pd.merge(df, df_basic, on=['ts_code', 'trade_date'], how='left')
df_merged = df_merged.dropna(subset=['pe', 'pb', 'total_mv'])
df_merged['eps'] = df_merged['close'] / df_merged['pe']
df_merged['circ_mv'] = df_merged['total_mv'] / df_merged['pb']
df_merged = df_merged.sort_values(by=['eps', 'circ_mv'], ascending=False)

# 分组收益率回测
group_num = 10
group_size = len(df_merged) // group_num

# 对数据进行标准化处理
X = df_merged[['eps', 'circ_mv']]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
y = df_merged['pct_chg'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = lr.predict(X_test)

# 输出 R2 分数
r2 = r2_score(y_test, y_pred)
print("R2 Score: {:.2f}".format(r2))

# 计算每个分组的收益率
df_merged['group'] = pd.cut(np.arange(len(df_merged)), bins=group_num, labels=False)
df_merged['group_return'] = df_merged.groupby(['group'])['pct_chg'].cumsum()

# 输出回测结果
group_return = df_merged.groupby(['group'])['group_return'].tail(1)
group_return.index = np.arange(1, group_num+1)
print(group_return)
