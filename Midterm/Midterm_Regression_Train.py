# 시작하기에 앞서 #########################
# train.csv 파일은 헤더를 제외하고 4464행으로 이루어짐
# → 3월 한 달간(31일) 10분 간격으로 ???를 조사한 결과로 예상됨
# test_set.csv 파일은 Excel 프로그램을 이용하여 정렬한 결과,
# 4/1 ~ 5/27 까지 랜덤한 시간대에서 ???를 조사한 내용(Target)을 추론해야한다는 결론.
# 
# <고려할 내용>
# → "주기성을 갖는 데이터"
# → "시계열 데이터 분석"

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

##################################################
# Prepare train dataset #########################
##################################################
df = pd.read_csv('C:/Users/USER/DKU_DL/train.csv')
# print(df.head())
# print(df.columns)

df['yymm'] = pd.to_datetime(df['yymm'], format='%m%d %H:%M') 
df_date = df['yymm'].dt.strftime('%m/%d %H:%M')
# print(df_date.head())


df_X = df.loc[:, (df.columns != 'Target') & (df.columns != 'yymm')]
df_X['V0'] = df['yymm'].dt.strftime('%M').astype('int')
# df_X.info()
df_y = df['Target']


# ##################################################
# # Analysis of dataset 
# ##################################################
# from matplotlib import pyplot as plt
# plt.plot(df_date[1500:1530], df_y[1500:1530])
# plt.show()
# # 그래프 분석 결과 1시간 간격으로 피크를 찍는? 것을 관찰 (최대/최소)
# # → 날짜와는 관계가 없어보이나, 20~40분 주기로 최대-최소 값이 변동됨
# # → 분(minute) 특성을 V0로 추가한다(?)


##################################################
# Split the data into training/testing sets 
##################################################
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.3, random_state=1234)

##################################################
# Evaluation indicators for regression
##################################################
Regression_Indicators = ['neg_mean_squared_error', 'r2']

##################################################    
# Support Vector Machine(SVM)_FS, Param Tuning Done!
##################################################
# print("[SVM Feature Selection]")
# model_SVM = svm.SVR(
#             # C=2.000,    # default= 1, 오류 허용 정도를 결정, 클 수록 오류 허용 안 함
#             kernel='rbf',
#             # gamma=0.0021, # default= 'scale', 데이터 각각의 영향력을 결정
#     )

# sfs_SVM = SequentialFeatureSelector(model_SVM, direction='forward', n_features_to_select='auto')

# fit_SVM = sfs_SVM.fit(df_X, df_y)
# print("Num Features: %d" % fit_SVM.n_features_in_)
# fs = df_X.columns[fit_SVM.support_].tolist()   # selected features
# print("Selected Features: %s" % fs)

# for i in Regression_Indicators:
#     scores = cross_val_score(model_SVM, df_X[fs], df_y, cv=5, scoring=i)
#     print(f"Acc({i}): {np.mean(scores):.4f}")

    # [SVM Feature Selection] → V26, auto
    # Num Features: 26
    # Selected Features: ['V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V10', 'V15', 'V16', 'V17', 'V21', 'V23', 'V25']
    # Acc(neg_mean_squared_error): -209.0340
    # Acc(r2): 0.0008

    # [SVM Feature Selection] → V0, auto
    # Num Features: 27
    # Selected Features: ['V1', 'V5', 'V8', 'V9', 'V13', 'V15', 'V16', 'V18', 'V19', 'V23', 'V24', 'V26', 'V0']
    # Acc(neg_mean_squared_error): -209.3857
    # Acc(r2): -0.0009

    # [SVM Feature Selection] → V0, 9 Features
    # Num Features: 27
    # Selected Features: ['V1', 'V5', 'V8', 'V9', 'V13', 'V16', 'V23', 'V24', 'V0']
    # Acc(neg_mean_squared_error): -209.2855
    # Acc(r2): -0.0004



# print("[SVM Param Tuning]")
# exponential_C_list = [-4, -3, -2, -1]
# exponential_gamma_list = [-7, -6, -5, -4, -3]
# params = {
#     'C': [2**i for i in exponential_C_list],
#     'gamma': [2**i for i in exponential_gamma_list]
# }

# model_SVM = svm.SVR()

# grid_search= GridSearchCV(model_SVM, params, cv=5, scoring='neg_mean_squared_error')
# fs=['V1', 'V5', 'V8', 'V9', 'V13', 'V16', 'V23', 'V24', 'V0']

# df_X_for_SVM = df_X.loc[:, fs]

# grid_search.fit(df_X_for_SVM, df_y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

    # [SVM Param Tuning] → V0, auto, Coarse Tuning
    # exponential_C_list = [-2, 0, 2, 4, 6]
    # exponential_gamma_list = [-4, -2, 0, 2, 4]
    # {'C': 0.25, 'gamma': 0.0625}
    # -209.21883789664403

    # [SVM Param Tuning] → V0, auto, Fine Tuning
    # exponential_C_list = [-4, -3, -2, -1]
    # exponential_gamma_list = [-7, -6, -5, -4, -3]
    # {'C': 0.125, 'gamma': 0.0625}
    # -209.21477124314515

    # [SVM Param Tuning] → V0, 9 Features
    # exponential_C_list = [-4, -3, -2, -1]
    # exponential_gamma_list = [-7, -6, -5, -4, -3]
    # {'C': 0.25, 'gamma': 0.125}
    # -209.2139500411658


##################################################
# XGBoost(XGB)_FS, Param Tuning Done!
##################################################
# print("[XGB Feature Selection]")
# model_XGB = xgb.XGBRegressor(
#     learning_rate=0.1,
#     n_estimators=1000,
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0,
#     objective='reg:squarederror',
#     )

# sfs_XGB = SequentialFeatureSelector(model_XGB, direction='forward', n_features_to_select='auto')

# fit_XGB = sfs_XGB.fit(df_X, df_y)
# print("Num Features: %d" % fit_XGB.n_features_in_)
# fs = df_X.columns[fit_XGB.support_].tolist()   # selected features
# print("Selected Features: %s" % fs)

# for i in Regression_Indicators:
#     scores = cross_val_score(model_XGB, df_X[fs], df_y, cv=5, scoring=i)
#     print(f"Acc({i}): {np.mean(scores):.4f}")

    # [XGB Feature Selection] → V26, auto
    # Num Features: 26
    # Selected Features: ['V2', 'V7', 'V8', 'V11', 'V12', 'V15', 'V17', 'V19', 'V20', 'V21', 'V23', 'V25', 'V26']
    # Acc(neg_mean_squared_error): -259.5197
    # Acc(r2): -0.2411
    
    # [XGB Feature Selection] → V0, auto
    # Num Features: 27
    # Selected Features: ['V2', 'V7', 'V8', 'V11', 'V12', 'V16', 'V17', 'V19', 'V21', 'V23', 'V25', 'V26', 'V0']
    # Acc(neg_mean_squared_error): -254.5707
    # Acc(r2): -0.2176

    # [XGB Feature Selection] → V0, 9 Features
    # Num Features: 27
    # Selected Features: ['V7', 'V8', 'V11', 'V12', 'V17', 'V19', 'V21', 'V25', 'V0']
    # Acc(neg_mean_squared_error): -253.0145
    # Acc(r2): -0.2098

# print("[XGB Param Tuning] → V0")

# D_train = xgb.DMatrix(train_X, label=train_y)
# D_test = xgb.DMatrix(test_X, label=test_y)

# fs = ['V7', 'V8', 'V11', 'V12', 'V17', 'V19', 'V21', 'V25', 'V0']
# fs.append('Target')
# df['yymm'] = pd.to_datetime(df['yymm'], format='%m%d %H:%M') 
# df['V0'] = df['yymm'].dt.strftime('%M').astype('int')
# df_for_XGB = df.loc[:, fs]
# df_X_for_XGB = df_for_XGB.loc[:,df_for_XGB.columns != 'Target']
# print(df_for_XGB.head())


    # XGBoost를 구성하는 트리에 대한 파라미터인 max_depth와 min_child_weight를 정해보자
# param_test1={
#     'max_depth': [3, 4, 5, 6, 7],
#     'min_child_weight': [6, 7, 8, 9, 10, 11, 12, 13]
# }

# grid_search1= GridSearchCV(XGBRegressor(
#                             learning_rate=0.1,
#                             n_estimators=100,
#                             # max_depth=5,
#                             # min_child_weight=1,
#                             gamma=0,
# ), param_test1, cv=5, scoring='neg_mean_squared_error')

# grid_search1.fit(df_X_for_XGB, df_y,)
# print(grid_search1.best_params_) # {'max_depth': 3, 'min_child_weight': 8}
# print(grid_search1.best_score_)  # -213.73475931843092


#     # 다음은 gamma에 대해 튜닝해보자
# param_test2={
#     'gamma': [0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
# }

# grid_search2= GridSearchCV(XGBRegressor(
#                             learning_rate=0.1,
#                             n_estimators=100,
#                             max_depth=3,
#                             min_child_weight=7,
#                             # gamma=0,
# ), param_test2, cv=5, scoring='neg_mean_squared_error')

# grid_search2.fit(df_X_for_XGB, df_y)
# print(grid_search2.best_params_) # {'gamma': 0}
# print(grid_search2.best_score_)  # -213.73475931843092


#     # 지금까지 파라미터 max_depth, min_child_weight, gamma 를 튜닝하였다. 지금 상태에서 최적의 n_estimators와 learning_rate를 다시 찾는다.
# param_test3={
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
#     'n_estimators': [100, 150, 200, 250, 300, 500, 700, 900, 1000]
# }
# grid_search3= GridSearchCV(XGBRegressor(
#                             # learning_rate=0.1,
#                             # n_estimators=100,
#                             max_depth=3,
#                             min_child_weight=7,
#                             gamma=0,
# ), param_test3, cv=5, scoring='neg_mean_squared_error')

# grid_search3.fit(df_X_for_XGB, df_y)
# print(grid_search3.best_params_) # {'learning_rate': 0.01, 'n_estimators': 100}
# print(grid_search3.best_score_)  # -209.07956549739419



    # [XGB Param Tuning] → V0, auto
    # params={
    #     'learning_rate'=0.01,
    #     'n_estimators'=100,
    #     'max_depth'=3,
    #     'min_child_weight'=8,
    #     'gamma'=0,
    #     'objective'='reg:squarederror',
    # }
    # -209.07956549739419

    # [XGB Param Tuning] → V0, auto
    # params={
    #     'learning_rate'=0.01,
    #     'n_estimators'=100,
    #     'max_depth'=3,
    #     'min_child_weight'=12,
    #     'gamma'=0,
    #     'objective'='reg:squarederror',
    # }
    # -209.04686427800607

    # [XGB Param Tuning] → V0, 9 Features
    # params={
    #     'learning_rate'=0.1,
    #     'n_estimators'=100,
    #     'max_depth'=3,
    #     'min_child_weight'=7,
    #     'gamma'=0,
    #     'objective'='reg:squarederror',
    # }
    # -208.96443420805946

##################################################
# Random Forest(RF)_FS, Param Tuning Done!
##################################################

# print("[RF Feature Selection]")
# model_RF = RandomForestRegressor(
#             n_estimators= 100,   # default= 100, 생성할 트리의 개수
#             criterion="absolute_error"     # default= "absolute_error", 평가 지표
#             # https://resultofeffort.tistory.com/107 참고
#             )

# sfs_RF = SequentialFeatureSelector(model_RF, direction='forward', n_features_to_select='auto')

# fit_RF = sfs_RF.fit(df_X, df_y)
# print("Num Features: %d" % fit_RF.n_features_in_)
# fs = df_X.columns[fit_RF.support_].tolist()   # selected features
# print("Selected Features: %s" % fs)

# for i in Regression_Indicators:
#     scores = cross_val_score(model_RF, df_X[fs], df_y, cv=5, scoring=i)
#     print(f"Acc({i}): {np.mean(scores):.4f}")
    
    # [RF Feature Selection] → V26, auto
    # Num Features: 26
    # Selected Features: ['V2', 'V3', 'V4', 'V5', 'V6', 'V8', 'V9', 'V11', 'V18', 'V21', 'V23', 'V24', 'V25']
    # Acc(neg_mean_squared_error): -211.8410
    # Acc(r2): -0.0122

    # [RF Feature Selection] → V0, auto
    # Num Features: 27
    # Selected Features: ['V1', 'V3', 'V4', 'V5', 'V6', 'V8', 'V11', 'V16', 'V18', 'V19', 'V21', 'V24', 'V0']
    # Acc(neg_mean_squared_error): -212.4816
    # Acc(r2): -0.0184

    # [RF Feature Selection] → V0, 9 Features
    # Num Features: 27
    # Selected Features: ['V3', 'V5', 'V8', 'V10', 'V11', 'V13', 'V21', 'V25', 'V0']
    # Acc(neg_mean_squared_error): -214.9468
    # Acc(r2): -0.0263

# print("[RF Param Tuning] → V0, 9 Features, Fine Tuning")
# fs = ['V3', 'V5', 'V8', 'V10', 'V11', 'V13', 'V21', 'V25', 'V0']
# df_X_for_RF = df_X.loc[:, fs]
# # print(df_X_for_RF.head())

# params= {
#     'n_estimators': [100],
#     'criterion':["absolute_error"],
#     'max_depth':[3, 4, 5, 6],
#     'max_features':[4, 5, 6, 7],
# }

# model_RF = RandomForestRegressor()

# grid_search = GridSearchCV(model_RF, params, cv=5, scoring='neg_mean_squared_error') 

# grid_search.fit(df_X_for_RF,df_y)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

    # [RF Param Tuning] → V26, auto, Coarse Tuning
    # params ={
    #     'n_estimators': [50, 100, 200, 500],
    #     'criterion':["absolute_error"],
    #     'max_depth':[4, 5, 6],
    #     'max_features':[4, 6, 8, 10, 12],
    # }
    # {'criterion': 'absolute_error', 'max_depth': 5, 'max_features': 8, 'n_estimators': 100}
    # -209.49069274090837

    # [RF Param Tuning] → V0, auto, Fine Tuning
    # params= {
    #     'n_estimators': [75, 100, 125, 150],
    #     'criterion':["absolute_error"],
    #     'max_depth':[4, 5, 6],
    #     'max_features':[6, 7, 8, 9, 10],
    # }
    # {'criterion': 'absolute_error', 'max_depth': 6, 'max_features': 7, 'n_estimators': 100}
    # -209.960937550685

    # [RF Param Tuning] → V0, 9 Features, Coarse Tuning
    # params= {
    #     'n_estimators': [75, 100, 125, 150],
    #     'criterion':["absolute_error"],
    #     'max_depth':[4, 5, 6],
    #     'max_features':[6, 7, 8, 9, 10],
    # }
    # {'criterion': 'absolute_error', 'max_depth': 4, 'max_features': 6, 'n_estimators': 100}
    # -210.2184025931163

    # [RF Param Tuning] → V0, 9 Features, Fine Tuning
    #     params= {
    #     'n_estimators': [100],
    #     'criterion':["absolute_error"],
    #     'max_depth':[3, 4, 5, 6],
    #     'max_features':[4, 5, 6, 7],
    # {'criterion': 'absolute_error', 'max_depth': 3, 'max_features': 4, 'n_estimators': 100}
    # -210.06141415060814