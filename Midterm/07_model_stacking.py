from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# prepare dataset
df_X, df_y  = load_breast_cancer(return_X_y=True) 

# define level 0
estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, 
                                   random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42, dual='auto')))]

# define model
model_1 = StackingClassifier(
      estimators=estimators, 
      final_estimator=LogisticRegression())

scores_1 = cross_val_score(model_1, df_X, df_y, cv=5)
print(np.mean(scores_1))

model_2 = StackingClassifier(
          estimators=estimators, 
          final_estimator=LogisticRegression(max_iter=1000),
          passthrough=True)

scores_2 = cross_val_score(model_2, df_X, df_y, cv=5)
print(np.mean(scores_2))

estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, 
                                   random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42, dual='auto'))),
     ('lr', LogisticRegression(max_iter=5000))]

from xgboost import XGBClassifier

model_3 = StackingClassifier(
         estimators=estimators, 
         final_estimator=XGBClassifier( 
                                    eval_metric='logloss', 
                                    random_state=42),
         passthrough=True)

scores_3 = cross_val_score(model_3, df_X, df_y, cv=5)
print(np.mean(scores_3))
