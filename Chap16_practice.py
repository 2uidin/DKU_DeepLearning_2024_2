import kagglehub

# Download latest version
path = kagglehub.dataset_download("szrlee/stock-time-series-20050101-to-20171231")

# print("Path to dataset files:", path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/SIM/DKU_DL/AABA_2006-01-01_to_2018-01-01.csv',
                #  skiprows=6, skipfooter=9,
                 engine='python')
df.head()

# 전처리
# from pandas.tseries.offsets import MonthEnd

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.head())
df.plot()

# 2017/6/30 까지의 데이터를 트레이닝셋.
# 그 이후 데이터를 테스트셋으로 한다.
# 예측할 feature는 Unadjusted

split_date = pd.Timestamp('2017-06-30')

train = df.loc[:split_date, ['Close']]
test = df.loc[split_date:, ['Close']]

ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], 
                           index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], 
                          index=test.index)
train_sc_df.head()

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = \
                       train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = \
                       test_sc_df['Scaled'].shift(s)

train_sc_df.head(13)

# NA 포함 행 제거
X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]

X_train.head()
X_test.head()

# dataframe to ndarray
X_train = X_train.values
X_test= X_test.values

y_train = y_train.values
y_test = y_test.values
print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)

# 최종 데이터셋 구성
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

print("Final DATASET")
print(X_train_t.shape)
print(X_train_t)
print(y_train)

# LSTM 모델 구축 ############################
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
#from keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential()              # Sequeatial Model
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature)
model.add(Dense(1))               # output = 1
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.fit(X_train_t, y_train, epochs=100,
          batch_size=30, verbose=1)

y_pred = model.predict(X_test_t)
print(y_pred)

from sklearn.metrics import mean_squared_error
print('Mean squared error: {0:.2f}'.\
      format(mean_squared_error(y_test, y_pred))) 

plt.figure()
plt.plot(y_pred)
plt.plot(y_test)
plt.legend(['predict', 'actual'])
plt.show()
