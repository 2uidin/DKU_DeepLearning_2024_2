# 필요한 모듈 불러오기
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 데이터 프레임 정의
df = pd.read_csv('C:/Users/SIM/DKU_DL/BostonHousing.csv')
# print(df)
df_X = df[['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad',\
    'tax', 'ptratio', 'b', 'lstat']]
df_y = df['medv']

# 데이터를 Training Data와 Test Data로 구분
train_X, test_X, train_y, test_y = \
train_test_split(df_X, df_y, test_size=0.3, random_state=1234)

# 학습 방식 정의
model = LinearRegression()
# Training Data를 이용하여 모델을 학습 → Q1
model.fit(train_X, train_y)
# Test Data를 이용하여 모델의 예측값 계산
pred_y = model.predict(test_X)
# print(pred_y)


# # 회귀 모델의 MSE 계산 → Q2
# print('Mean squared error: {0:.2f}'.\
#     format(mean_squared_error(test_y, pred_y)))
# # 회귀 모델의 R^2 score 계산 → Q2
# print('Coefficient of determination: %.2f'
#     % r2_score(test_y, pred_y))

# 회귀 모델의 계수와 절편 계산 → Q3
print('Coefficients:')
for i in range(len(model.coef_)):
    print('{:.2f}'.format(model.coef_[i]))
print('Intercept: {0:3f}'.format(model.intercept_)) 