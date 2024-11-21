# 필요한 lib import

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 데이터셋 불러오기
df = pd.read_csv('C:/Users/SIM/DKU_DL/PimaIndiansDiabetes.csv')
# print(df.head())    
# print(df.columns)

# 독립변수(X)와 종속변수(y) 설정
df_X = df.loc[:, df.columns != 'diabetes']
df_y = df['diabetes']

# xgboost의 학습 과정에서 데이터셋에 문자열이 포함되어있는 경우에는 학습이 불가능하므로,
# 미리 neg=0, pos=1로 변환
number=LabelEncoder()
labeled_df_y=number.fit_transform(df_y).astype('int')

# print(df_y)
# print(labeled_df_y)

# Training/Testing set으로 데이터셋 분할
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, labeled_df_y, test_size=0.3,\
                     random_state=1234)

### DT ###
model_DT = DecisionTreeClassifier(\
            # max_depth=5,\
            criterion="entropy",\
            random_state=1234,\
            min_samples_leaf=20,\
            # min_samples_split=30\
            )

model_DT.fit(train_X, train_y)

# performance evaluation
print("[Algorithm: Decision Tree]")
print('Train accuracy : {:.4f}'.format(model_DT.score(train_X, train_y)))
print('Test accuracy : {:.4f}'.format(model_DT.score(test_X, test_y)))

# pred_y = model_DT.predict(test_X)
# print(confusion_matrix(test_y, pred_y))

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# plot_tree(model_DT, 
#           fontsize=8, 
#           feature_names=df_X.columns.to_list(),
#           class_names=True)
# plt.show()
## DT ###

### SVM ###
model_SVM = svm.SVC(
            C=2.000,\
            kernel='rbf',\
            gamma=0.0021\
    )

model_SVM.fit(train_X, train_y)

# performance evaluation
print("[Algorithm: Support Vector Machine]")
print('Train accuracy : {:.4f}'.format(model_SVM.score(train_X, train_y)))
print('Test accuracy : {:.4f}'.format(model_SVM.score(test_X, test_y)))

# pred_y = model_SVM.predict(test_X)
# print(confusion_matrix(test_y, pred_y))
### SVM ###

### RF ###
model_RF = RandomForestClassifier(
            n_estimators=75,\
            # max_depth=4,\
            criterion="entropy",\
            random_state=1234,\
            # min_samples_leaf=12,\
            min_samples_split=19            
            )

model_RF.fit(train_X, train_y)

# performance evaluation
print("[Algorithm: Random Forest]")
print('Train accuracy : {:.4f}'.format(model_RF.score(train_X, train_y)))
print('Test accuracy : {:.4f}'.format(model_RF.score(test_X, test_y)))

# pred_y = model_RF.predict(test_X)
# print(confusion_matrix(test_y, pred_y))
### RF ###

### xgboost ###
D_train = xgb.DMatrix(train_X, label=train_y)
D_test = xgb.DMatrix(test_X, label=test_y)

param = {
    'eta': 0.15,
    'max_depth': 5, 
    'objective': 'binary:logistic',
    'eval_metric': 'error'}
 
steps = 18

model_xgb = xgb.train(param, D_train, steps)

# performance evaluation
print("[Algorithm: xgboost]")

train_pred = model_xgb.predict(D_train)  
train_round_preds = np.round(train_pred) # real -> [0,1] 
print('Train accuracy : {:.4f}'.format(accuracy_score(train_y, train_round_preds)))

test_pred = model_xgb.predict(D_test)  
test_round_preds = np.round(test_pred) # real -> [0,1] 
print('Test accuracy : {:.4f}'.format(accuracy_score(test_y, test_round_preds)))
### xgboost ###