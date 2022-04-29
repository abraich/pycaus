import pandas as pd
import numpy as np


# import data from csv
#df = pd.read_csv('data/gite.csv')
# normale data simulation
N =1
Y_0 = np.random.normal(loc=0, scale=10, size=N)
Y_1 = np.random.normal(loc=0, scale=10, size=N)

CATE = Y_1 - Y_0

def gite(y_1, y_0,u):
    return np.exp(-u*y_1) - np.exp(-u*y_0)
U = np.linspace(0,1,10)
GITE = []
for u in U:
    GITE.append(gite(Y_1, Y_0, u)[0])
print(GITE)
print(CATE)