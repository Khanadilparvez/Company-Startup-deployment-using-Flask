# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:44:36 2019

@author: ADIL KHAN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startups.csv')


X = dataset.iloc[:, :4]
X=X.drop(['Marketing Spend'],axis=1)

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'New York':1, 'California':2, 'Florida':3
                }
    return word_dict[word]

X['State'] = X['State'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('Linearmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('Linearmodel.pkl','rb'))
print(model.predict([[100000, 80000,3]]))
