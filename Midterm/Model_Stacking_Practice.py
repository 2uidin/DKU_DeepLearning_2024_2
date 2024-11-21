# basemodel 종류,
# Regression
# └linear regression(Simple, Multiple)
# └logistic regression → 범주형 데이터를 회귀로 해결

# Classification
# └DT
# └RF
# └XGB
# └SVM

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# prepare dataset
df = pd.read_csv('C:/Users/SIM/DKU_DL/madelon.csv')
df_X = df.loc[:, df.columns != 'class']
df_y = df['class']

# xgboost의 학습 과정에서 데이터셋에 문자열이 포함되어있는 경우에는 학습이 불가능하므로,
# 미리 neg=0, pos=1로 변환
number=LabelEncoder()
labeled_df_y=number.fit_transform(df_y).astype('int')

# Training/Testing set으로 데이터셋 분할
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, labeled_df_y, test_size=0.3,\
                     random_state=1234)
    
########################################
# Decision Tree
########################################
model_DT = DecisionTreeClassifier(
            criterion="entropy",
            random_state=1234,
            min_samples_leaf=20,
            )    
model_DT.fit(train_X, train_y)

print("[Model: Decision Tree]")
print("Accuracy: %.3f" % model_DT.score(train_X, train_y))

########################################
# Support Vector Machine
########################################
model_SVM = svm.SVC(
            C=2.000,
            kernel='rbf',
            gamma=0.0021
            )
model_SVM.fit(train_X, train_y)

print("[Model: Support Vector Machine]")
print("Accuracy: %.3f" % model_SVM.score(test_X, test_y))

########################################
# Random Forest
########################################
model_RF = RandomForestClassifier(
            n_estimators=75,
            criterion="entropy",
            random_state=1234,
            min_samples_split=20        
            )
model_RF.fit(train_X, train_y)

print("[Model: Random Forest]")
print("Accuracy: %.3f" % model_RF.score(test_X, test_y))

########################################
# XGBoost
########################################
model_XGB = XGBClassifier(eta=0.15,
            max_depth=5,
            objective='binary:logistic',
            eval_metric='error')
model_XGB.fit(train_X, train_y)

print("[Model: XGBoost]")
print("Accuracy: %.3f" % model_XGB.score(test_X, test_y))


# define level 0
estimators = [
    ('dt', DecisionTreeClassifier(
            criterion="entropy",
            random_state=1234,
            min_samples_leaf=20,
            )),
    ('svm', svm.SVC(
            C=2.000,
            kernel='rbf',
            gamma=0.0021
            )),
     ('rf', RandomForestClassifier(
            n_estimators=75,
            criterion="entropy",
            random_state=1234,
            min_samples_split=20        
            )),
     ('xgb', XGBClassifier(eta=0.15,
            max_depth=5,
            objective='binary:logistic',
            eval_metric='error'))
    #  ('svr', make_pipeline(StandardScaler(),
    #                        LinearSVC(random_state=42, dual='auto')))
     ]

# define model
model = StackingClassifier(
          estimators=estimators, 
          final_estimator=LogisticRegression(solver='saga', max_iter=10000),
          passthrough=True)

scores = cross_val_score(model, df_X, labeled_df_y, cv=5)
print("Model Stacking Accurarcy: %.3f" % scores.mean())
print(scores.mean())


 