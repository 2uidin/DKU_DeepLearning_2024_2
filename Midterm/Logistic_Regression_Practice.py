# 필요한 모듈 불러오기
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터셋 불러오기
X, y = datasets.load_wine(return_X_y=True)
# print(X.shape)
# print(y)

# 데이터를 Training Data와 Test Data로 구분
train_X, test_X, train_y, test_y = \
    train_test_split(X, y, test_size=0.3, random_state=1234) 

# 학습 방식 정의
model = LogisticRegression()
# Training Data를 이용하여 모델을 학습
model.fit(train_X, train_y)
# Test Data를 이용하여 모델의 예측값 계산 → Q1
pred_y = model.predict(test_X)
# print(pred_y)


# 모델의 예측 정확도 출력 → Q2
acc_for_train_X = accuracy_score(train_y, model.predict(train_X))
print('Accuracy for train set : {0:3f}'.format(acc_for_train_X))
acc_for_test_X = accuracy_score(test_y, pred_y)
print('Accuracy for test set : {0:3f}'.format(acc_for_test_X))