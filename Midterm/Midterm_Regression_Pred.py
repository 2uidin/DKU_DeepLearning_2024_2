from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
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
# print(df.head())
# print(df.columns)

df['yymm'] = pd.to_datetime(df['yymm'], format='%m%d %H:%M') 
df_date = df['yymm'].dt.strftime('%m/%d %H:%M')
# print(df_date.head())


df_X = df.loc[:, (df.columns != 'Target') & (df.columns != 'yymm')]
df_X['V0'] = df['yymm'].dt.strftime('%M').astype('int')
# df_X.info()
df_y = df['Target']

##################################################
# Prepare target dataset 
##################################################
df_pred = pd.read_csv('C:/Users/SIM/DKU_DL/test_set.csv')
df_pred['yymm'] = pd.to_datetime(df_pred['yymm'], format='%m%d %H:%M') 
df_pred['V0'] = df_pred['yymm'].dt.strftime('%M').astype('int')

##################################################    
# Support Vector Machine(SVM)_FS, Param Tuning Done!
##################################################
model_SVM = svm.SVR()
params_SVM = {
    'kernel': 'rbf',
    'C': 0.18,
    'gamma': 0.0624, 
}
model_SVM.set_params(**params_SVM)

fs_for_SVM = ['V1', 'V5', 'V8', 'V9', 'V13', 'V15', 'V16', 'V18', 'V19', 'V23', 'V24', 'V26', 'V0']
df_X_for_SVM = df_X.loc[:, fs_for_SVM]
model_SVM.fit(df_X_for_SVM, df_y)

# pred_SVM = model_SVM.predict(df_pred.loc[:, fs_for_SVM])

# pred_SVM_df = pd.DataFrame(np.round(pred_SVM, decimals=4), columns=['predict'])    # 배열을 df로 변환

# pred_SVM_df.to_csv('C:/Users/USER/Desktop/32192406_심의진.csv', index=False)   # 인덱스 열을 제외하고 csv파일로 저장

##################################################
# Random Forest(RF)_FS, Param Tuning Done!
##################################################
model_RF = RandomForestRegressor()
params_RF = {
    'n_estimators': 100,
    'max_depth': 6,
    'max_features': 7,
    'criterion':"absolute_error",
}
model_RF.set_params(**params_RF)

fs_for_RF = ['V1', 'V3', 'V4', 'V5', 'V6', 'V8', 'V11', 'V16', 'V18', 'V19', 'V21', 'V24', 'V0']
df_X_for_RF = df_X.loc[:, fs_for_RF]
model_RF.fit(df_X_for_RF, df_y)

# pred_RF = model_RF.predict(df_pred.loc[:, fs_for_RF])

# pred_RF_df = pd.DataFrame(np.round(pred_RF, decimals=4), columns=['predict'])

# pred_RF_df.to_csv('C:/Users/USER/Desktop/32192406_심의진.csv', index=False)

##################################################
# XGBoost(XGB)_FS, Param Tuning Done!
##################################################
model_XGB = xgb.XGBRegressor()
params_XGB = {
    'learning_rate': 0.01,
    'n_estimators': 100,
    'max_depth': 3,
    'min_child_weight': 12,
    'gamma': 0,
    'objective': 'reg:squarederror',
}
model_XGB.set_params(**params_XGB)

fs_for_XGB = ['V2', 'V7', 'V8', 'V11', 'V12', 'V16', 'V17', 'V19', 'V21', 'V23', 'V25', 'V26', 'V0']
df_X_for_XGB = df_X.loc[:, fs_for_XGB]
model_XGB.fit(df_X_for_XGB, df_y)


# pred_XGB = model_XGB.predict(df_pred.loc[:, fs_for_XGB])

# pred_XGB_df = pd.DataFrame(np.round(pred_XGB, decimals=4), columns=['predict'])

# pred_XGB_df.to_csv('C:/Users/SIM/Desktop/32192406_심의진.csv', index=False)

##################################################
# Voting Regression
##################################################
models = [
    ('svm', model_SVM),
    ('xgb', model_XGB),
    ('rf', model_RF),
]
voting_regressor = VotingRegressor(models)

fs_for_Voting = ['V3', 'V5', 'V7', 'V8', 'V9', 'V11', 'V15', 'V18', 'V19', 'V20', 'V25', 'V26', 'V0']
df_X_for_Voting = df_X.loc[:, fs_for_Voting]
voting_regressor.fit(df_X_for_Voting, df_y)

pred_Voting = voting_regressor.predict(df_pred.loc[:, fs_for_Voting])

pred_XGB_df = pd.DataFrame(np.round(pred_Voting, decimals=4), columns=['predict'])

pred_XGB_df.to_csv('C:/Users/SIM/Desktop/32192406_심의진.csv', index=False)

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

# fs_for_Stacking = ['V3', 'V5', 'V7', 'V8', 'V9', 'V11', 'V15', 'V18', 'V19', 'V20', 'V25', 'V26', 'V0']
# df_X_for_Voting = df_X.loc[:, fs_for_Stacking]
# stacking_regressor.fit(df_X_for_Voting, df_y)

# pred_Voting = stacking_regressor.predict(df_pred.loc[:, fs_for_Stacking])

# pred_XGB_df = pd.DataFrame(np.round(pred_Voting, decimals=4), columns=['predict'])

# pred_XGB_df.to_csv('C:/Users/SIM/Desktop/32192406_심의진.csv', index=False)

'''
[1st Trial]                         [acc]_neg_mean_squared_error
    Model: SVM
    Feature ver.: V0, auto          -209.3857
    Param ver.:  V0, auto, Fine     -209.21477124314515
    mae: 12.61

[2nd Trial]
    Model: RF
    Feature ver.: V26, auto         -211.8410
    Param ver.:  V26, auto, Course  -209.49069274090837
    mae: 12.66

[3rd Trial]
    Model: RF
    Feature ver.: V0, auto          -212.4816
    Param ver.:  V0, auto, Fine     -209.960937550685
    mae: 12.72

[4th Trial]
    Model: XGB
    Feature ver.: V0, auto          -254.5707
    Param ver.: V0, auto            -209.07956549739419
    mae: 12.63

[5th Trial]
    Model: XGB
    Feature ver.: V0, 9 Features    -253.0145
    Param ver.: V0, 9 Features      -208.96443420805946
    mae: 12.61
'''
'''
[6th Trial]
    Model: SVM
    Feature ver.: V26, auto         -209.0340
    Param ver.: None                -(-)
    mae: 12.67

[7th Trial]
    Model: SVM
    Feature ver.: V0, auto          -209.3857
    Param ver.:  V0, auto, Fine     -209.21477124314515
    mae: 12.61

[8th Trial]
    Model: Voting(Opt_SVM, Opt_XGB, Opt_RF)
    Feature ver.: V0, auto          -209.2273
    Param ver.: None                -(-)
    mae: 12.59

[9th Trial]
    Model: Voting(Opt_SVM, Opt_XGB, Opt_RF)
    Feature ver.: None              -209.1788
    Param ver.: None                -(-)
    mae: 12.60
    
[10th Trial]
    Model: XGB
    Feature ver.: V0, auto          -254.5707
    Param ver.: V0, auto            -209.04686427800607
    mae: 12.61
'''
'''
[11st Trial]
    Model: Voting(9 Features: Opt_SVM, Opt_XGB, Opt_RF)
    Feature ver.: None              -209.1788
    Param ver.: None                -(-)
    mae: 12.61
'''