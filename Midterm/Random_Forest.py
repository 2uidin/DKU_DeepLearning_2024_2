# Random Forest Example


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# prepare the credit dataset
df = pd.read_csv('C:/Users/SIM/DKU_DL/liver.csv')
print(df.head())    
print(df.columns)   # column names

df_X = df.loc[:, df.columns != 'category']
df_y = df['category']

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.3,\
                     random_state=1234) 

# Define learning model (# of tree: 10)  #####################################
model = RandomForestClassifier(n_estimators=10, random_state=1234)
# Train the model using the training sets
model.fit(train_X, train_y)

# performance evaluation
print('Train accuracy :', model.score(train_X, train_y))
print('Test accuracy :', model.score(test_X, test_y))

pred_y = model.predict(test_X)
confusion_matrix(test_y, pred_y)

# Define learning model ((# of tree: 50)  ####################################
model = RandomForestClassifier(n_estimators=50, random_state=1234)

# Train the model using the training sets
model.fit(train_X, train_y)

# performance evaluation
print('Train accuracy :', model.score(train_X, train_y))
print('Test accuracy :', model.score(test_X, test_y))

pred_y = model.predict(test_X)
confusion_matrix(test_y, pred_y)

# visualize tree
# Extract single tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

estimator = model.estimators_[5]     # 6th tree
plot_tree(estimator, 
          fontsize=6, 
          feature_names=df_X.columns.to_list(),
          class_names=True)
plt.show()