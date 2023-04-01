from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import tushare as ts
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

pro = ts.pro_api(token='2cb7b30105056801542e9ff02a8370c509b84397686c94376801c170')
hs300 = pro.index_weight(index_code='000300.SH', start_date='20220301', end_date='20220331')

df = pd.DataFrame()
for ts_code in hs300.con_code.tolist():
    data = pro.daily(ts_code=ts_code, start_date='20220301', end_date='20220331')
    df = pd.concat([df, data], ignore_index=True)

df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values(by='trade_date', ascending=True)

df_basic = pro.daily_basic(ts_code=','.join(hs300.con_code.tolist()), trade_date='20220331',
                            fields='ts_code,trade_date,pe,pb,total_mv')

df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])

df_merged = pd.merge(df, df_basic, on=['ts_code', 'trade_date'], how='left')
df_merged = df_merged.dropna(subset=['pe', 'pb', 'total_mv'])
df_merged['eps'] = df_merged['close'] / df_merged['pe']
df_merged['circ_mv'] = df_merged['total_mv'] / df_merged['pb']
df_merged = df_merged.sort_values(by=['eps', 'circ_mv'], ascending=False)

print(df_merged)

group_num = 10
group_size = len(df_merged) // group_num

# 将时间序列数据转换为输入特征
def time_series_to_features(df, time_steps):
    features = []
    for i in range(time_steps, len(df)):
        features.append(df[i-time_steps:i, 0])
    return np.array(features)

# 对数据进行标准化处理
X = df_merged[['eps', 'circ_mv']].values
scaler = StandardScaler()
X = scaler.fit_transform(X[:, [0]])

# 将数据转换为时间序列形式
time_steps = 30
X_ts = time_series_to_features(X, time_steps)
y_ts = df_merged['pct_chg'].values[time_steps:]

# 划分训练集和测试集
train_size = int(len(X_ts) * 0.8)
X_train, X_test = X_ts[:train_size], X_ts[train_size:]
y_train, y_test = y_ts[:train_size], y_ts[train_size:]

# 建立 RNN 模型
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_ts.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 将输入数据reshape成LSTM模型需要的形状
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# 可视化训练集和验证集上的损失变化
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # 预测测试集数据
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred).ravel()

# 预测结果
print('预测结果y_pre=',sep=' ',end=' ')
print(y_pred)

# 进行测试
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# 计算R方值
r2 = r2_score(y_test, y_pred)
print('R方值:', r2)

# 计算每组收益率
df_merged['y_pred'] = np.nan
df_merged.iloc[-len(y_pred):, df_merged.columns.get_loc('y_pred')] = y_pred.reshape(1, -1)[0]
df_merged['group'] = pd.qcut(df_merged['eps']*df_merged['circ_mv'], group_num, labels=False)
grouped = df_merged.groupby('group')
group_ret = grouped['pct_chg'].mean()

# 可视化回测结果
plt.plot(group_ret.index, group_ret.values)
plt.title('Grouped Returns')
plt.xlabel('Group')
plt.ylabel('Average Return')
plt.show()

# 可视化模型预测结果的分布
sns.histplot(y_pred, kde=True)
plt.title('Distribution of predicted values')
plt.xlabel('Percentage change')
plt.show()

# 可视化预测结果和真实值的对比
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Test set: True vs predicted')
plt.xlabel('Index')
plt.ylabel('Percentage change')
plt.legend()
plt.show()

# 可视化预测结果与输入特征之间的关系
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].scatter(X_test[:, 0], y_test, alpha=0.5)
axs[0].scatter(X_test[:, 0], y_pred, alpha=0.5)
axs[0].set_xlabel('EPS')
axs[0].set_ylabel('Percentage change')
axs[0].legend(['True', 'Predicted'])
axs[1].scatter(X_test[:, 1], y_test, alpha=0.5)
axs[1].scatter(X_test[:, 1], y_pred, alpha=0.5)
axs[1].set_xlabel('Circulating market value')
axs[1].set_ylabel('Percentage change')
axs[1].legend(['True', 'Predicted'])
plt.show()
