# xgboost Example


import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# prepare the credit dataset
df = pd.read_csv('C:/Users/SIM/DKU_DL/liver.csv')

df_X = df.loc[:, df.columns != 'category']
df_y = df['category'] 

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.3,\
                     random_state=1234) 

D_train = xgb.DMatrix(train_X, label=train_y)
D_test = xgb.DMatrix(test_X, label=test_y)

# Define & train learning model  #####################################
param = {
    'eta': 0.2, 
    'max_depth': 3,  
    'objective': 'binary:logistic',
    'eval_metric': 'error'}
 
steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)

pred = model.predict(D_test)  
round_preds = np.round(pred) # real -> [0,1] 

from sklearn.metrics import accuracy_score
accuracy_score(test_y, round_preds)

