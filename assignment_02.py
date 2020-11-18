#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:45:18 2020

@author: kimjong-gyu
"""


import pandas as pd 
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


house=pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
house.columns
house_r=house.drop(['id','date','zipcode'],axis=1)

columns=house_r.columns

features=""

for i in columns:
    if i =='price':
        features+=""
    else:
        features+=i+"+"

features=features[:-1]
 
y,X=dmatrices('price~'+features,house_r,return_type='dataframe')   

#calculate vif
vif=pd.DataFrame()
vif['VIF Factor']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"]=X.columns
print(vif)

#vif가 inf인 sqft_living
X_1=X.drop(['sqft_living'],axis=1)
vif_1=pd.DataFrame()
vif_1['VIF Factor']=[variance_inflation_factor(X_1.values, i) for i in range(X_1.shape[1])]
vif_1["features"]=X_1.columns
print(vif_1)

#delete 'Intercept' because this value is highest 
X_2=X_1.drop(['Intercept'],axis=1)
vif_2=pd.DataFrame()
vif_2['VIF Factor']=[variance_inflation_factor(X_2.values, i) for i in range(X_2.shape[1])]
vif_2["features"]=X_2.columns
print(vif_2)

#delete 'long' because this value is highest 
X_3=X_2.drop(['long'],axis=1)
vif_3=pd.DataFrame()
vif_3['VIF Factor']=[variance_inflation_factor(X_3.values, i) for i in range(X_3.shape[1])]
vif_3["features"]=X_3.columns
print(vif_3)

#delete 'yr_built' because this value is highest 
X_4=X_3.drop(['yr_built'],axis=1)
vif_4=pd.DataFrame()
vif_4['VIF Factor']=[variance_inflation_factor(X_4.values, i) for i in range(X_4.shape[1])]
vif_4["features"]=X_4.columns
print(vif_4)

#delete 'grade' because this value is highest 
X_5=X_4.drop(['grade'],axis=1)
vif_5=pd.DataFrame()
vif_5['VIF Factor']=[variance_inflation_factor(X_5.values, i) for i in range(X_5.shape[1])]
vif_5["features"]=X_5.columns
print(vif_5)

#delete 'lat' because this value is highest 
X_6=X_5.drop(['lat'],axis=1)
vif_6=pd.DataFrame()
vif_6['VIF Factor']=[variance_inflation_factor(X_6.values, i) for i in range(X_6.shape[1])]
vif_6["features"]=X_6.columns
print(vif_6)

#delete 'bathrooms' because this value is highest 
X_7=X_6.drop(['bathrooms'],axis=1)
vif_7=pd.DataFrame()
vif_7['VIF Factor']=[variance_inflation_factor(X_7.values, i) for i in range(X_7.shape[1])]
vif_7["features"]=X_7.columns
print(vif_7)

#delete 'sqft_living15' because this value is highest 
X_8=X_7.drop(['sqft_living15'],axis=1)
vif_8=pd.DataFrame()
vif_8['VIF Factor']=[variance_inflation_factor(X_8.values, i) for i in range(X_8.shape[1])]
vif_8["features"]=X_8.columns
print(vif_8)

#delete 'bedrooms' because this value is highest 
X_9=X_8.drop(['bedrooms'],axis=1)
vif_9=pd.DataFrame()
vif_9['VIF Factor']=[variance_inflation_factor(X_9.values, i) for i in range(X_9.shape[1])]
vif_9["features"]=X_9.columns
print(vif_9)

#delete 'floors' because this value is highest 
X_10=X_9.drop(['floors'],axis=1)
vif_10=pd.DataFrame()
vif_10['VIF Factor']=[variance_inflation_factor(X_10.values, i) for i in range(X_10.shape[1])]
vif_10["features"]=X_10.columns
print(vif_10)

#delete corr value>0.5
corr=house[X_10.columns].corr()>0.5
corr
#final value 
lm=sm.OLS(house_r['price'],house_r[['sqft_lot','waterfront','view','condition','sqft_above','sqft_basement','yr_renovated']])
result=lm.fit()
result.summary()
print(result.summary())







