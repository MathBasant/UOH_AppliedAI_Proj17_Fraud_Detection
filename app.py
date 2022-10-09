# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:58:05 2022

@author: leobu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
#from sklearn.externals
import joblib
# import pickle

app = Flask(__name__)

#model = pickle.load(open('model.pkl', 'rb'))
clf = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #load test dataset
    test =  pd.read_csv("testf.csv")
    
    #call standardscaler and fit to test data
    standard = StandardScaler()
    test_sc = standard.fit_transform(test)
    
    # onvert input str to int and use it as index to test_sc
    # for x in request.form.values()
    x = request.form.values()
    # int_index = int(x) 
    prediction = clf.predict([test_sc[x]])

    if prediction[0] == 0 :
        output = 'Legit (0)'
    else :
        output = 'Fraudulent (1)'

    return render_template('index.html', prediction_text='Transaction is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)