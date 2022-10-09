# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:22:39 2022

@author: leobu
"""

# import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix , classification_report , f1_score, roc_auc_score

import pandas as pd
pd.set_option('display.max_columns', 50)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#from sklearn.externals
import joblib

# we don't have to fill missing values etc since the dataset in this folder is already preprocessed
# the datasets need to be scaled or normalised only 
# train vs test is already split : two separate datasets

# load train data
df = pd.read_csv('trainf.csv')

# split into x and y
x = df.drop(columns=['isFraud'], axis=1)
y = df['isFraud']

# Standardize data in x
standard = StandardScaler()
x_sc = standard.fit_transform(x)

# Call model from XGBoost with parameters determined in ipynb 
clf = XGBClassifier(n_estimators = 200 , reg_lambda = 10**-4 )

#Fitting model with trainig data
clf.fit(x_sc, y)

# Saving model to disk
joblib.dump(clf, 'model.pkl')
# pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))