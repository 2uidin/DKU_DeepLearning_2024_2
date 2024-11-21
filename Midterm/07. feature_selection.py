# Feature selection Example

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# prepare the dataset
df = pd.read_csv('C:/Users/SIM/DKU_DL/PimaIndiansDiabetes.csv')
print(df.head())    
print(df.columns)   # column names

df_X = df.loc[:, df.columns != 'diabetes']
df_y = df['diabetes']

# whole features
model = LogisticRegression(solver='lbfgs', max_iter=3000)
scores = cross_val_score(model, df_X, df_y, cv=5)
print("Acc: "+str(scores.mean()))   # 모든 Feature를 사용했을 경우의 Acc

######################################################################
# feature selection by filter method
######################################################################
# feature evaluation method : chi-square
from sklearn.feature_selection import SelectKBest   # Filter Method를 기본으로 진행하는 알고리즘
from sklearn.feature_selection import chi2          # 평가 척도로 Chi^2를 이용한다

test = SelectKBest(score_func=chi2, k=df_X.shape[1])
fit = test.fit(df_X, df_y)
# summarize evaluation scores
print(np.round(fit.scores_, 3)) # Feature의 평가 점수 출력

f_order = np.argsort(-fit.scores_)  # sort index by decreasing order
sorted_columns = df.columns[f_order]

# test classification accuracy by selected features (KNN)
model = LogisticRegression(solver='lbfgs', max_iter=500)
for i in range(1, df_X.shape[1]+1):
    fs = sorted_columns[0:i]
    df_X_selected = df_X[fs]
    scores = cross_val_score(model, df_X_selected, df_y, cv=5)
    print(fs.tolist())
    print(np.round(scores.mean(), 4))
# 평가 점수가 좋은 순서부터 하나씩 넣어봄

######################################################################
# Forward Search
######################################################################
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(solver='lbfgs', max_iter=500)

sfs = SequentialFeatureSelector(model, direction='forward', n_features_to_select='auto')

fit = sfs.fit(df_X, df_y)
print("Num Features: %d" % fit.n_features_in_)
fs = df_X.columns[fit.support_].tolist()   # selected features
print("Selected Features: %s" % fs)

scores = cross_val_score(model, df_X[fs], df_y, cv=5)
print("Acc: "+str(scores.mean()))

######################################################################
# Backward elimination (Recursive Feature Elimination)
######################################################################
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='lbfgs', max_iter=500)
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(df_X, df_y)
print("Num Features: %d" % fit.n_features_)
fs = df_X.columns[fit.support_].tolist()   # selected features
print("Selected Features: %s" % fs)
#print("Feature Ranking: %s" % fit.ranking_)

scores = cross_val_score(model, df_X[fs], df_y, cv=5)
print("Acc: "+str(scores.mean()))

'''
######################################################################
# Different Forward selection 
######################################################################
# please install 'mlxtend' moudle  

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

model = LogisticRegression(solver='lbfgs', max_iter=500)
sfs1 = SFS(model, 
           k_features=5, 
           verbose=2,
           scoring='accuracy',
           cv=5)

sfs1 = sfs1.fit(df_X, df_y)
# numpy 라이브러리 제거 후 ver.1.26.4 설치 → pip install numpy==1.26.4
# scikit-learn 라이브러리 제거 후 ver.1.4.2 설치 → 미해결
sfs1.subsets_             # selection process
sfs1.k_feature_idx_       # selected feature index  
sfs1.k_feature_names_     # selected feature name

scores = cross_val_score(model, df_X[list(sfs1.k_feature_names_)], df_y, cv=5)
print("Acc: "+str(scores.mean()))
'''