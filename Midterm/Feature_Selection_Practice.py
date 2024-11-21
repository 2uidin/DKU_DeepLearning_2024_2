import pandas as pd
import numpy as np  # 현재 ver.1.26.4 /2024-10-09 작성
from sklearn.linear_model import LogisticRegression # 현재 ver.1.4.2 /2024-10-09 작성
from sklearn.model_selection import cross_val_score

# 데이터셋 준비
df = pd.read_csv('C:/Users/SIM/DKU_DL/madelon.csv')
# print(df.head())    
# print(df.columns)   # column names

df_X = df.loc[:, df.columns != 'class']
df_y = df['class']

######################################################################
# feature selection by filter method
######################################################################
# feature evaluation method : chi-square
from sklearn.feature_selection import SelectKBest   # Filter Method를 기본으로 진행하는 알고리즘
from sklearn.feature_selection import chi2          # 평가 척도로 Chi^2를 이용한다

test = SelectKBest(score_func=chi2, k=df_X.shape[1])
fit = test.fit(df_X, df_y)
# summarize evaluation scores
# print(np.round(fit.scores_, 3)) # Feature의 평가 점수 출력

f_order = np.argsort(-fit.scores_)  # sort index by decreasing order
sorted_columns = df.columns[f_order]
# print(sorted_columns)

# test classification accuracy by selected features (KNN)
model = LogisticRegression(solver='lbfgs', max_iter=500)
for i in range(1, 11):  # 상위 10개의 feature만 고려
    fs = sorted_columns[0:i]
    df_X_selected = df_X[fs]
    scores = cross_val_score(model, df_X_selected, df_y, cv=5)
    print("Selected Features(Filter Method): %s" % fs.tolist())
    print("Acc(Filter Method): "+str(np.round(scores.mean(), 3)))
# 평가 점수가 좋은 순서부터 하나씩 넣어봄


######################################################################
# Forward Search
######################################################################
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(solver='lbfgs', max_iter=500)

sfs = SequentialFeatureSelector(model, direction='forward', n_features_to_select=4)
# Filter Method의 결과 4개의 feature를 통해서 acc를 도출했으므로,
# 같은 feature를 사용하여 더 높은 acc를 얻는 것을 목표로 함.
fit = sfs.fit(df_X, df_y)
print("Num Features(SFS): %d" % fit.n_features_in_)
fs = df_X.columns[fit.support_].tolist()   # selected features
print("Selected Features(SFS): %s" % fs)

scores = cross_val_score(model, df_X[fs], df_y, cv=5)
print("Acc(SFS): "+str(scores.mean())+"\n")


######################################################################
# Backward elimination (Recursive Feature Elimination)
######################################################################
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest   # Filter Method를 기본으로 진행하는 알고리즘
from sklearn.feature_selection import chi2          # 평가 척도로 Chi^2를 이용한다

model = LogisticRegression(solver='lbfgs', max_iter=500)
rfe = RFE(model, n_features_to_select=4)    # Iteration Error 발생, 각 feature의 acc점수가 높은 상위 10개만 사용

test = SelectKBest(score_func=chi2, k=df_X.shape[1])
fit = test.fit(df_X, df_y)
f_order = np.argsort(-fit.scores_)
sorted_columns = df.columns[f_order]
fs = sorted_columns[0:10]
# print("Selected Features(RFE): %s" % fs.tolist())

fit = rfe.fit(df_X[fs], df_y)

print("Num Features(RFE): %d" % fit.n_features_)
fs = df_X[fs].columns[fit.support_].tolist()   # selected features
print("Selected Features(RFE): %s" % fs)
# print("Feature Ranking: %s" % fit.ranking_)

scores = cross_val_score(model, df_X[fs], df_y, cv=5)
print("Acc(RFE): "+str(scores.mean())+"\n")