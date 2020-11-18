#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 01:06:36 2020

@author: kimjong-gyu
"""


import pandas as pd
import numpy as np

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

#문이열려있는데 판매량이 0인 데이터 제거
train2=train[(train['Open']==1)&(train['Sales']>0)]

train_store = pd.merge(train2, store, how = 'inner', on = 'Store')
train_store['SalePerCustomer']=train_store['Sales']/train_store['Customers']
train_store['StateHoliday'][train_store['StateHoliday']!=0 ] = 1
train_store=train_store.drop(['Promo2SinceWeek','Promo2SinceYear'],axis=1)


train_store.fillna(0)

#prom2 and promo2Interval
#make category
train_store['PromoInterval'].value_counts()
conditionlist = [
    (train_store['PromoInterval'] =='Jan,Apr,Jul,Oct') ,
    (train_store['PromoInterval'] =='Feb,May,Aug,Nov'),
    (train_store['PromoInterval'] =='Mar,Jun,Sept,Dec')]

choicelist = [1,2,3]
train_store['PI'] = np.select(conditionlist, choicelist, default=0)

train_store['PI'].value_counts()
train_store=train_store.drop(['Promo2','PromoInterval'],axis=1)

#variables competition distance 
train_store['CompetitionDistance'].value_counts()
mid=train_store['CompetitionDistance'].mean()
f=0.5*mid
t=1.5*mid
conditionlist = [
    (train_store['CompetitionDistance'] <= t) & (train_store['CompetitionDistance'] !=0),
    (train_store['CompetitionDistance'] <= mid) & (train_store['CompetitionDistance']>t),
    (train_store['CompetitionDistance'] <= f) & (train_store['CompetitionDistance'] >mid),
    (train_store['CompetitionDistance'] >= f),
    (train_store['CompetitionDistance']==0)]

choicelist = [4,3,2,1,0]
train_store['CD'] = np.select(conditionlist, choicelist, default=0)
train_store['CD'].value_counts()
train_store=train_store.drop(['CompetitionDistance'],axis=1)

#CompetitionOpenSince[Month/Year]
train_store['CompetitionOpenSinceYear'].value_counts()
train_store['Date']=pd.to_datetime(train_store['Date'])
train_store['Year']=train_store['Date'].dt.year
train_store['Month']=train_store['Date'].dt.month
train_store['CompetitionOpen']=12*(train_store.Year-train_store.CompetitionOpenSinceYear)+train_store.Month-train_store.CompetitionOpenSinceMonth
train_store['CompetitionOpen'].value_counts()

mid1=train_store['CompetitionOpen'].mean()
f1=0.5*mid
t1=1.5*mid
conditionlist = [
    (train_store['CompetitionOpen'] <= t1)&(train_store['CompetitionOpen'] != 0),
    (train_store['CompetitionOpen'] <= mid1) & (train_store['CompetitionOpen'] >t1),
    (train_store['CompetitionOpen'] <= f1) & (train_store['CompetitionOpen'] >mid1),
    (train_store['CompetitionOpen'] >f1),
    (train_store['CompetitionOpen']==0)]

choicelist = [4,3,2,1,0]
train_store['CO'] = np.select(conditionlist, choicelist, default=0)
train_store['CO'].value_counts()
train_store=train_store.drop(['CompetitionOpen'],axis=1)
train_store=train_store.drop(['CompetitionOpenSinceYear','CompetitionOpenSinceMonth'],axis=1)

#historical

train_store=train_store.set_index('Date')
new_variables=train_store.groupby('Store')['Sales'].rolling(window='7D').mean().to_frame().rename(columns={'Sales':'Sales1W'})
new_variables['Sales2W']=train_store.groupby('Store')['Sales'].rolling(window='14D').mean()

new_variables['Sales1_2_diff']=new_variables['Sales1W']-new_variables['Sales2W']

new_variables['Sales1_2_ratio']=new_variables['Sales1W']/new_variables['Sales2W']

new_variables.head(30)

new_variables=new_variables.reset_index()
new_variables['Date']=new_variables['Date']+pd.to_timedelta('7D')


new_variables.head()

new_train_store=train_store.merge(new_variables, on=['Store','Date'],how='left')
#2주부터 구할려고 하기 때문에
new_train_store=new_train_store[new_train_store['Date']>=pd.to_datetime('2013-01-15')]
new_train_store=new_train_store.fillna(0)

final_train=new_train_store.drop(['Customers'],axis=1)

final_train=final_train.drop(['Year','Month','Store'],axis=1)

#make dummy data 

catvar=['DayOfWeek','PI','StoreType','Assortment','CO','CD']
for c in catvar:
    temp=pd.get_dummies(final_train[c],prefix=c, drop_first=True)
    final_train=pd.concat((final_train,temp),axis=1)
    
final_train=final_train.drop(catvar,axis=1)
final_train=final_train[final_train['Open']==1]

train_value=final_train[final_train['Date']<=pd.to_datetime('20141231')]
test_value=final_train[final_train['Date']>pd.to_datetime('20141231')]

trainY=train_value['Sales']
trainX=train_value.drop(['Sales','Open','Date'],axis=1)

valY=test_value['Sales']
valX=test_value.drop(['Sales','Open','Date'],axis=1)

#linear
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold , StratifiedKFold, GroupKFold


reg1=LinearRegression()
reg1.fit(trainX,trainY)
reg1.coef_
reg1.score(trainX,trainY)
linear_r2=reg1.score(valX,valY)
linear_r2

reg2=Ridge(alpha=1)
reg2.fit(trainX,trainY)
reg2.coef_

reg3=Lasso(alpha=1)
reg3.fit(trainX,trainY)
reg3.coef_

alphas=np.logspace(-3,3,30)

#R2 squared its not depend alpha
# whole model 
linear_r2=reg1.score(valX,valY)

result=pd.DataFrame(index=alphas,columns=['Ridge','Lasso'])
for alpha in alphas:
    reg2.alpha=alpha
    reg3.alpha=alpha
    reg2.fit(trainX,trainY)
    result.loc[alpha,'Ridge']=reg2.score(valX,valY)
    reg3.fit(trainX,trainY)
    result.loc[alpha,'Lasso']=reg3.score(valX,valY)


plt.plot(np.log(alphas),result['Ridge'],label='Ridge')
plt.plot(np.log(alphas),result['Lasso'],label='Lasso')
plt.hlines(linear_r2,np.log(alphas[0]),np.log(alphas[-1]),ls=':',colors='k',label='Ordinary')
plt.legend()
print(result)  

w_ridge=13.738238
w_lasso=0.788046 

#k-fold model
alphas=np.logspace(-3,3,30)
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet,ElasticNetCV

ridge_cv=RidgeCV(alphas=alphas, cv=5)
model = ridge_cv.fit(trainX,trainY)
print(model.alpha_)

k_ridge=148.73521072935117

lasso_cv=LassoCV(alphas=alphas, cv=5)
model2 = lasso_cv.fit(trainX,trainY)
print(model2.alpha_)

k_lasso=0.3039195382313198


#final calculation from whole train
final_reg1_e=Ridge(alpha=w_ridge)
final_reg2_e=Lasso(alpha=w_lasso)

final_reg1_e.fit(trainX,trainY)
final_reg1_e.score(trainX,trainY)
final_reg1_e.score(valX,valY)

final_reg2_e.fit(trainX,trainY)
final_reg2_e.score(trainX,trainY)
final_reg2_e.score(valX,valY)


#from kfold
final_reg3_e=Ridge(alpha=k_ridge)
final_reg4_e=Lasso(alpha=k_lasso)

final_reg3_e.fit(trainX,trainY)
final_reg3_e.score(trainX,trainY)
final_reg3_e.score(valX,valY)

final_reg4_e.fit(trainX,trainY)
final_reg4_e.score(trainX,trainY)
final_reg4_e.score(valX,valY)













