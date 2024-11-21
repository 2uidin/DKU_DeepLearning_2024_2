from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
import pandas as pd
import numpy as np

##################################################
# Prepare train dataset
##################################################
df = pd.read_csv('C:/Users/SIM/DKU_DL/train.csv')

df['yymm'] = pd.to_datetime(df['yymm'], format='%m%d %H:%M') 
df_date = df['yymm'].dt.strftime('%m/%d %H:%M')

df_X = df.loc[:, (df.columns != 'Target') & (df.columns != 'yymm')]
df_X['V0'] = df['yymm'].dt.strftime('%M').astype('int')

df_y = df['Target']

##################################################
# Split the data into training/testing sets 
##################################################
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.3, random_state=1234)

##################################################
# Single Model Definition
##################################################

    # SVM
model_SVM = svm.SVR()
params_SVM = {
    'kernel': 'rbf',
    'C': 0.25,
    'gamma': 0.125,
}
model_SVM.set_params(**params_SVM)

fs_for_SVM = ['V1', 'V5', 'V8', 'V9', 'V13', 'V16', 'V23', 'V24', 'V0']
df_X_for_SVM = df_X.loc[:, fs_for_SVM]
model_SVM.fit(df_X_for_SVM, df_y)

    # XGB
model_XGB = xgb.XGBRegressor()
params_XGB = {
'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 3,
    'min_child_weight': 7,
    'gamma': 0,
    'objective': 'reg:squarederror',
}
model_XGB.set_params(**params_XGB)

fs_for_XGB = ['V7', 'V8', 'V11', 'V12', 'V17', 'V19', 'V21', 'V25', 'V0']
df_X_for_XGB = df_X.loc[:, fs_for_XGB]
model_XGB.fit(df_X_for_XGB, df_y)

    # RF
model_RF = RandomForestRegressor()
params_RF = {
    'n_estimators': 100,
    'max_depth': 3,
    'max_features': 4,
    'criterion':"absolute_error",
}
model_RF.set_params(**params_RF)

fs_for_RF = ['V3', 'V5', 'V8', 'V10', 'V11', 'V13', 'V21', 'V25', 'V0']
df_X_for_RF = df_X.loc[:, fs_for_RF]
model_RF.fit(df_X_for_RF, df_y)

##################################################
# Ensemble model definition
##################################################
models = [
    ('svm', model_SVM),
    ('xgb', model_XGB),
    ('rf', model_RF),
]

##################################################
# define Feature Selector
##################################################
def FeatureSelection (ensemble_type):
    train_X, test_X, train_y, test_y = \
        train_test_split(df_X, df_y, test_size=0.3, random_state=1234)
    
    ensemble_type.fit(train_X, train_y)
    ensemble_pred = ensemble_type.predict(test_X)

    print(f'[Evaluation before Feature Selection] → Optimal Single Model(9 Features, auto), All Feature Ensemble')
    print(f'Stacking 분류기 정확도: {np.mean(cross_val_score(ensemble_type, df_X, df_y, cv=5, scoring='neg_mean_squared_error')):.4f}')

    for i in models:
        i[1].fit(train_X, train_y)
        pred = i[1].predict(test_X)
        print(f'{i[0]} 정확도: {np.mean(cross_val_score(i[1], df_X, df_y, scoring='neg_mean_squared_error')):.4f}')

    print("/########################################/")

    sfs_Ensemble = SequentialFeatureSelector(ensemble_type, direction='forward', n_features_to_select= 9)
    fit_sfs_Ensemble = sfs_Ensemble.fit(df_X, df_y)
    print(f'[Stacking FS Result]')
    print("Num Features: %d" % fit_sfs_Ensemble.n_features_in_)
    fs = df_X.columns[fit_sfs_Ensemble.support_].tolist()   # selected features
    print("Selected Features: %s" % fs)

    print("/########################################/")
    
    sfs_Ensemble = SequentialFeatureSelector(ensemble_type, direction='forward', n_features_to_select='auto')
    fit_sfs_Ensemble = sfs_Ensemble.fit(df_X, df_y)
    print(f'[Stacking FS Result]')
    print("Num Features: %d" % fit_sfs_Ensemble.n_features_in_)
    fs = df_X.columns[fit_sfs_Ensemble.support_].tolist()   # selected features
    print("Selected Features: %s" % fs)

    print("/########################################/")

    df_X_for_Ensemble = df_X.loc[:, fs]
    train_X, test_X, train_y, test_y = \
        train_test_split(df_X_for_Ensemble, df_y, test_size=0.3, random_state=1234)

    ensemble_type.fit(train_X, train_y)
    ensemble_pred = ensemble_type.predict(test_X)

    print("[Evaluation after Feature Selection] → Optimal Single Model(9 Features, auto), Selected Feature Ensemble")
    print(f'Stacking 분류기 정확도: {np.mean(cross_val_score(ensemble_type, df_X_for_Ensemble, df_y, cv=5, scoring='neg_mean_squared_error')):.4f}')

    for i in models:
        i[1].fit(train_X, train_y)
        pred = i[1].predict(test_X)
        print(f'{i[0]} 정확도: {np.mean(cross_val_score(i[1], df_X_for_Ensemble, df_y, scoring='neg_mean_squared_error')):.4f}')


##################################################
# Voting Regression
##################################################
voting_regressor = VotingRegressor(models)

FeatureSelection(voting_regressor)

##################################################
# Stacking Regression
##################################################
# stacking_regressor = StackingRegressor(
#                         estimators= models,
#                         )
# params_stacking_regressor ={
#     'estimators': models,
#     'cv': 5,
#     'final_estimator': LinearRegression(n_jobs=-1),
#     'passthrough': True,

# }
# stacking_regressor.set_params(**params_stacking_regressor)

# FeatureSelection(stacking_regressor)

    # [Voting Acc Evaluation] → Pure State
    # Voting 분류기 정확도: -217.0867
    # svm 정확도: -209.4397
    # xgb 정확도: -271.9298
    # rf 정확도: -213.5301

    # [Voting Acc Evaluation] → Optimal Single Model(V0, auto), All Feature Ensemble
    # Voting 분류기 정확도: -209.1788
    # svm 정확도: -209.4343
    # xgb 정확도: -209.1623
    # rf 정확도: -210.7784

    # [Acc Evaluation] → Optimal Single Model(V0, auto), Selected Feature Ensemble
    # Voting 분류기 정확도: -209.2273
    # svm 정확도: -209.2742
    # xgb 정확도: -208.8410
    # rf 정확도: -210.2465

    # [Voting Feature Selection] → Optimal Single Model(V0, auto)
    # Num Features: 27
    # Selected Features: ['V3', 'V5', 'V7', 'V8', 'V9', 'V11', 'V15', 'V18', 'V19', 'V20', 'V25', 'V26', 'V0']






    # [Evaluation before Feature Selection] → Optimal Single Model(V0, auto), All Feature Ensemble
    # Stacking 분류기 정확도: -213.0127
    # svm 정확도: -209.4343
    # xgb 정확도: -209.1623
    # rf 정확도: -210.7766
##################################################
# Gradient Boost
##################################################

"""
일단 이거 돌아가는 지 보고,
1. 보팅회귀 모델에 대한 fs 진행
2. 보팅 모델 자체에도 n_estimater 같은 파라미터가 있네...?


+) 대략적인 로드맵..?
1. 사실 세 가지 앙상블 기법을 사용하려고 했다. (Voting, Bagging, Stacking)
그런데 아무래도 시간상 Bagging은 진행하기 힘들 듯.

2. 그래서 각 앙상블 모델에 대한 튜닝을 진행하되.
    1) 튜닝되지 않은 단일 모델을 이용하여 전체 특성으로 학습한 앙상블 모델
    2) 최적의 단일 모델을 이용하여 전체 특성으로 학습한 앙상블 모델 → 이 단계에서 앙상블 모델에 대한 Feature Selection 진행
    3) 최적의 단일 모델을 이용하여 2)의 특성으로 학습한 앙상블 모델
    4) 2)의 Feature Selection에 대한 파라미터 튜닝을 진행한 단일 모델을 이용하여 전체 특성으로 학습한 앙상블 모델
    5) 2)의 Feature Selection에 대한 파라미터 튜닝을 진행한 단일 모델을 이용하여 2)의 특성으로 학습한 앙상블 모델
    
의 접근 방식이 있을 것 같은데, 이 중에서
1)을 통해서 비교군 설정, 2)를 통해서 FS를 진행하고, 3)으로 마무리하는 게 나을 듯
4)는 시간 되면 한 번 해보는 걸로 하고, 5)는 단일모델의 튜닝이 의미가 있을까.. 싶음

3. 그렇다면 단일 모델에 대한 '최적'의 정의가 이루어져야 하는데,
우선, 단일모델은 전부 V0, auto 버전 특성을 주고, 파라미터 튜닝은 점수가 가장 좋은 거로.
그 다음 시간이 남으면 각 모델 별로 점수가 가장 좋았던 걸로 진행.
"""