# 07 Voting Classifier

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# load dataset
df_X, df_y  = load_breast_cancer(return_X_y=True)

# scaling data
sc_data = StandardScaler().fit(df_X)
df_X = sc_data.transform(df_X) 

# Define single models
clf_lr = LogisticRegression()
clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_dt = DecisionTreeClassifier(random_state=1)

# Define voting classifer
clf_voting = VotingClassifier(
              estimators=[('LR', clf_lr),
                          ('KNN', clf_knn),
                          ('DT', clf_dt)],
              voting='soft')

## Test each classifier ##########################
models = [clf_lr, clf_knn, clf_dt]
for model in models:
    scores = cross_val_score(model, df_X, df_y, cv=5)
    model_name = model.__class__.__name__
    print(f"{model_name} \t : {np.mean(scores)}") 

## Test Voting classifier ########################### 
voting_scores = cross_val_score(clf_voting, df_X, df_y, cv=5)
print('Voting accuracy', np.mean(voting_scores))


