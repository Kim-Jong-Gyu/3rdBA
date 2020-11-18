#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 01:41:46 2020

@author: kimjong-gyu
"""



import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway



house=pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
house.columns
house=house.drop(['id','zipcode','lat','long'],axis=1)


#EDA
#I divide into 6 category
binary=['waterfront']
Sqft=['sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15']
Date=['date','yr_built','yr_renovated']
In_condition=['bedrooms','bathrooms','floors']
cat=['view','condition','grade']

#Scatter matrix about Sqft
price='price'
Sqft.append(price)
scatter_matrix(house[Sqft],figsize=(12,8))


#visualize correlationship about Sqft
corr=house[Sqft].corr()
colormap = plt.cm.PuBu 
plt.figure(figsize=(10,8))
sns.heatmap(corr, linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})
Sqft.remove('sqft_living15')
Sqft.remove('sqft_above')
Sqft.remove('sqft_lot15')
house_s=house.drop(['sqft_living15','sqft_above','sqft_lot15'],axis=1)

#date variable --> binary type
var_split =house['date'].str.split("T")
value_name = var_split.str[0]
yr_c=value_name.str[0:4]
yr_c=yr_c.astype('int')
Date.remove('date')

#Scatter matrix about Date
Date.append(price)
scatter_matrix(house[Date],figsize=(12,8))

#visualize correlationship about Date
corr1=house[Date].corr()
colormap = plt.cm.PuBu 
plt.figure(figsize=(10,8))
sns.heatmap(corr1, linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {'size' : 16})

house_d=house_s.drop(['date','yr_built'],axis=1)


#Scatter matrix about In_condtion
In_condition.append(price)
scatter_matrix(house[In_condition],figsize=(12,8))


#visualize correlationship In_condition
corr1=house[In_condition].corr()
colormap = plt.cm.PuBu 
plt.figure(figsize=(10,8))
sns.heatmap(corr1, linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})

house_i=house_d.drop(['floors'],axis=1)


#waterfront 
varname='waterfront'
gmean=house.groupby(varname)[price].mean()
gstd=house.groupby(varname)[price].std()
plt.bar(range(len(gmean)),gmean)
plt.errorbar(range(len(gmean)),gmean,yerr=gstd,fmt='o',c='r',ecolor='r',capthick=2,capsize=3)
plt.xticks(range(len(gmean)),gmean.index)

#date
house_i['yr_c']=yr_c
varname='yr_c'
gmean1=house_i.groupby(varname)[price].mean()
gstd=house_i.groupby(varname)[price].std()
plt.bar(range(len(gmean1)),gmean1)
plt.errorbar(range(len(gmean1)),gmean1,yerr=gstd,fmt='o',c='r',ecolor='r',capthick=2,capsize=3)
plt.xticks(range(len(gmean1)),gmean1.index)


#VIF-> remainder(date&sqft&In_condition)
house_v=house_i.drop(['view','condition','grade'],axis=1)

#Binary 
bi=['waterfront','yr_c']
for c in bi:
    dummy=pd.get_dummies(house_v[c],prefix=c,drop_first=True)
    house_v=pd.concat((house_v,dummy),axis=1)
X2=house_v.drop(bi+['price'],axis=1)
y=house['price']

#VIF test 
y_v,X_v=dmatrices('price~'+'bedrooms+bathrooms+sqft_living+sqft_lot+sqft_basement+yr_renovated+waterfront_1+yr_c_2015',house_v,return_type='dataframe')   
vif=pd.DataFrame()

vif['VIF Factor']=[variance_inflation_factor(X_v.values, i) for i in range(X_v.shape[1])]
vif["features"]=X_v.columns
print(vif)

X2=sm.add_constant(X2)
model=sm.OLS(y,X2)
result=model.fit()
result.summary()

#Add Cat
house_v['view']=house['view']
house_v['grade']=house['grade']
house_v['condition']=house['condition']

#OLS + cat

catvar=['view','grade','condition']
#make dummy variable each categorical variable
for c in catvar:
    dummy=pd.get_dummies(house_v[c],prefix=c,drop_first=True)
    #this dummy variable is added to original 
    #combine 
    house_v=pd.concat((house_v,dummy),axis=1)
# original model
X3=house_v.drop(catvar+['price'],axis=1)
y3=house_v['price']
X3=sm.add_constant(X3)
model=sm.OLS(y3,X3)
result=model.fit()
result.summary()








