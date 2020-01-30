# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:57:51 2020

@author: naif
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Book3.csv')
X=dataset.iloc[:,[2,3,4,5,6,7,8,9,10]].values
y=dataset.iloc[:,[11]].values
#print(X)
#print(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0]= le.fit_transform(X[:,0])
#print(X[:,0])
X[:,1]= le.fit_transform(X[:,1])
#print(X[:,1])
X[:,7]= le.fit_transform(X[:,7])
#print(X[:,7])
X[:,8]= le.fit_transform(X[:,8])
#print(X[:,8])
#print(X)
#print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#print(x_train)

from sklearn.ensemble import RandomForestRegressor
reg =  RandomForestRegressor(n_estimators = 5000, random_state = 0)
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test) 
#print(y_pred)
from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
print(rms)
r2_score=r2_score(y_test,y_pred)
print(r2_score)
