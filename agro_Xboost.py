# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:22:39 2020

@author: naif
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
xgdmat=xgb.DMatrix(X_train,y_train)
our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}
final_gb=xgb.train(our_params,xgdmat)
tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb.predict(tesdmat)
#print(y_pred)

from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
print(rms)
r2_score=r2_score(y_test,y_pred)
print(r2_score)
# Applying k-Fold Cross Validation
