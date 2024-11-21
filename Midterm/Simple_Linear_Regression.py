# load required modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# prepare dataset
cars = pd.read_csv('C:/Users/SIM/DKU_DL/cars.csv')  #cars.csv 경로
print(cars)

speed = cars['speed'].to_frame()
dist= cars['dist']

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
train_test_split(speed, dist, test_size=0.2, random_state=123)
# test_size: 데이터 중 테스트 데이터로 사용할 비율
# random_state: 데이터를 랜덤하게 배열하는 시드
# → 데이터가 임의로 split 되므로 같은 시드를 부여하여 항상 같은 결과가 나오도록 함.

# Define learning method
model = LinearRegression()
# Train the model using the training sets
model.fit(train_X, train_y)
# Make predictions using the testing set
pred_y = model.predict(test_X)
print("pred_y")
print(pred_y)


# prediction test
print(model.predict([[13]])) # when speed=13
print(model.predict([[20]])) # when speed=20

# The coefficients & Intercept
print('Coefficients: {0:.2f}, Intercept {1:.3f}'\
.format(model.coef_[0], model.intercept_))
# model.coef_: 모델의 회귀 계수(기울기)
# model.intercept_: 모델의 절편

# The mean squared error
print('Mean squared error: {0:.2f}'.\
format(mean_squared_error(test_y, pred_y)))
# 실제 값과 모델에 의한 예측값의 차의 제곱의 합의 평균, 값이 작을수록 정확

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(test_y, pred_y))
# R^2 score: 1에 가까울수록 정확한 모델


# Plot outputs
plt.scatter(test_X, test_y, color='black')
plt.plot(test_X, pred_y, color='blue', linewidth=3)
plt.xlabel('speed')
plt.ylabel('dist')
plt.show()