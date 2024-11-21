# 07 Bagging Meta Classifier

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# load dataset
df_X, df_y  = load_breast_cancer(return_X_y=True)

# scaling data
sc_data = StandardScaler().fit(df_X)
df_X = sc_data.transform(df_X) 

# evaluate base model
model_base = KNeighborsClassifier()
scores = cross_val_score(model_base, df_X, df_y, cv=5)
print(np.mean(scores))

# bagging predictor
model_bagging = BaggingClassifier(KNeighborsClassifier(),
                             n_estimators = 100,       # 모델 수
                             max_samples=0.5,          # 인스턴스 선택 비율
                             max_features=1.0,         # feature 선택 비율
                             n_jobs=-1,
                             random_state=123)

scores_bagging = cross_val_score(model_bagging, df_X, df_y, cv=5)
print(np.mean(scores_bagging))
