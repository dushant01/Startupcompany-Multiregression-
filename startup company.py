# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:58:15 2020

@author: jethi
"""
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

""" importhing dat set """

data= pd.read_csv("50_Startups.csv")

X= data.iloc[:,:-1].values
Y= data.iloc[:,4].values
print(X)

""" Enconding the categorical varriable """

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
CT= ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[X]),('LE',LabelEncoder(),[X])],remainder='passthrough')
X= np.ndarray(3,3)(CT.fit_transform(X))


