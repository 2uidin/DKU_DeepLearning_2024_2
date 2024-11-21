# Simple Delta Rule

import numpy as np

x = np.array([0.5, 0.8, 0.2])
w = np.array([0.4, 0.7, 0.8])
d = 1
alpha = 0.5

def SIGMOID(x):
    return 1/(1 + np.exp(-x))

# update w
for i in range(50):
    v = np.sum(w*x)
    y = SIGMOID(v)
    e = d-y
    print("error", i, e)
    w = w + alpha*y*(1-y)*e*x