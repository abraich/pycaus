import pandas as pd
import numpy as np


# import data from csv
#df = pd.read_csv('data/gite.csv')
# normale data simulation
Y_0 = np.random.normal(loc=0, scale=10, size=10)
Y_1 = np.random.normal(loc=0, scale=10, size=10)

CATE = Y_1 - Y_0

def gite(y_1, y_0,u):
    return np.exp(-u*y_1) - np.exp(-u*y_0)
print(Y_0)
print(Y_1)
print(CATE)
print(gite(Y_1, Y_0, 0.5))