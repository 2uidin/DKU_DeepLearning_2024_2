# load required modules

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the prestge dataset
df = pd.read_csv('C:/Users/SIM/DKU_DL/prestige.csv')
print(df)
df_X = df[['education','women','prestige']]
df_y = df['income']
# 교육년수, 여성비율, 평판을 통해 직군의 연봉을 예측하는 과정
# → 중선형 회귀 함수의 x, y 설정
# print(df_X)
# print(df_y)

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
train_test_split(df_X, df_y, test_size=0.2,
random_state=123)
# Dfine learning model
model = LinearRegression()
# Train the model using the training sets
model.fit(train_X, train_y)
# Make predictions using the testing set
pred_y = model.predict(test_X)
print(pred_y)


# The coefficients & Intercept
print('Coefficients: W_1: {0:.2f}, W_2: {1:.2f}, W_3: {2:.2f}\nIntercept: {3:.3f}'\
      .format(model.coef_[0], model.coef_[1], model.coef_[2],model.intercept_))
# The mean squared error
print('Mean squared error: {0:.2f}'.\
format(mean_squared_error(test_y, pred_y)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(test_y, pred_y))


# Test single data
my_test_x = np.array([11.44,8.13,54.1]).reshape(1,-1)   # 1행 맞춤열의 데이터프레임 생성
my_pred_y = model.predict(my_test_x)
print(my_pred_y)