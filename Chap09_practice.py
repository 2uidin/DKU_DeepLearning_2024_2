# 이거 집에 빨리 가려고 복붙한 거니까 나중에 하나씩 뜯어봐야함 #


from sklearn import datasets
import random
import numpy as np
# SIGMOID function #####################################
def SIGMOID(x):
    return 1/(1 + np.exp(-x))
# SLP function #########################################
def SLP_SGD(tr_X, tr_y, alpha, rep):
    #initialize w
    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1,100), n)
    w = (np.array(w)-50)/100
    w = w.reshape(tr_X.shape[1],-1)
    # update w
    for i in range(rep):
        for k in range(tr_X.shape[0]):
            x = tr_X[k,:]
            v = np.matmul(x, w)
            y = SIGMOID(v)
            e = tr_y[k,:] - y
            w = w + alpha * np.matmul(tr_X[k,:].reshape(-1,1),
(y*(1-y)*e).reshape(1,-1))
        print("error", i, np.mean(e))
    return w
## prepare dataset #####################################
iris = datasets.load_iris()
X = iris.data
target = iris.target
# one hot encoding
num = np.unique(target, axis=0)
num = num.shape[0]
y = np.eye(num)[target]
## Training (get W) ####################################
W = SLP_SGD(X, y, alpha=0.01, rep=1000) 

