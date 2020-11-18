#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:35:17 2020

@author: kimjong-gyu
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, KFold



elec=pd.read_csv('https://drive.google.com/uc?export=download&id=1fq9qDqXLiUm0un_saxAUpPsSJa05F_bV', index_col=0)
county=pd.read_csv('https://drive.google.com/uc?export=download&id=1LciKFXkb3MmpXFEHDk1Db8YFsK0liF3a')

data=elec.merge(county,left_on='FIPS',right_on='fips',how='left')

data=elec[elec['county_name']!='Alaska'].merge(county,left_on='FIPS',right_on='fips',how='left')
data_ak=elec[elec['county_name']=='Alaska'].drop_duplicates(['votes_dem_2016','votes_gop_2016'])
data_ak['FIPS']=2000
data_ak=data_ak.merge(county,left_on='FIPS',right_on='fips',how= 'left')
data=pd.concat((data,data_ak),axis=0).sort_values('fips') 

#공화당과 민주당 
data['target']=(data['votes_dem_2016']>data['votes_gop_2016'])*1

target_value=data['target']
# only for using the county variables, i delete elect variable except target

rem_list=['combined_fips', 'votes_dem_2016', 'votes_gop_2016', 'total_votes_2016',
       'per_dem_2016', 'per_gop_2016', 'diff_2016', 'per_point_diff_2016',
       'state_abbr', 'county_name', 'FIPS', 'total_votes_2012',
       'votes_dem_2012', 'votes_gop_2012', 'county_fips', 'state_fips',
       'per_dem_2012', 'per_gop_2012', 'diff_2012', 'per_point_diff_2012']

data=data.drop(rem_list,axis=1)

#fifs and String type delete 
data=data.drop(['fips', 'area_name','state_abbreviation'], axis=1)

data.columns

# select 3 type RHI BZA EDU 
sel=['RHI225214','RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214',
     'RHI825214','BZA010213', 'BZA110213', 'BZA115213','EDU635213','EDU685213']
data_sel=data[sel]

# data scailing 
# white or not white 

race=['RHI225214','RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214','RHI825214']


# Standard scalling

# RHI825214->White alone, not Hispanic or Latino, percent, 2014
# that is not white race is 100 - RHI825214
# not white race
data_sel['RHI_nw']=100-data_sel['RHI825214']
data_sel=data_sel.drop(['RHI225214','RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214'],axis=1)

#emplyment and establishment correlation 
import matplotlib.pyplot as plt 
import seaborn as sns  
data_temp=data_sel[['BZA110213','BZA010213','BZA115213']]

plt.figure(figsize=(15,15))
sns.heatmap(data = data_temp.corr(), annot=True,fmt = '.2f', linewidths=.5, cmap='Blues')

# so I delete eastablishment 
data_sel=data_sel.drop(['BZA010213'],axis=1)

sdscal=StandardScaler()
sdscal.fit(data_sel)
sdscal_data=sdscal.transform(data_sel)
sdscal_data_f=pd.DataFrame(sdscal_data,columns=data_sel.columns)

sdscal_data_f.columns
#training set 
first=sdscal_data_f[['RHI825214','RHI_nw','BZA110213', 'BZA115213']]
second=sdscal_data_f[['BZA110213', 'BZA115213', 'EDU635213','EDU685213']]
third=sdscal_data_f[['RHI825214','RHI_nw','EDU635213','EDU685213']]

#train different model
#first 
clf1=LogisticRegression(C=1, max_iter=1000)
y1=data['target']
X1=first
k_fold = KFold(n_splits=5, random_state=1, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score': make_scorer(roc_auc_score)
          }

result1 = cross_validate(clf1, X1, y1, cv=k_fold, scoring=scoring)
accuracy1 = result1["test_accuracy"].mean()
precision1 = result1["test_precision"].mean()
recall1 = result1["test_recall"].mean()
f1_score1 = result1["test_f1_score"].mean()
auc_score1 = result1["test_roc_auc_score"].mean()

print("accuracy: {0: .4f}".format(accuracy1))
print("precision: {0: .4f}".format(precision1))
print("recall: {0: .4f}".format(recall1))
print("f1_score: {0: .4f}".format(f1_score1))
print("auc_score: {0: .4f}".format(auc_score1))

#second 
y2=data['target']
X2=second
k_fold = KFold(n_splits=5, random_state=1, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score': make_scorer(roc_auc_score)
          }

result2 = cross_validate(clf1, X2, y2, cv=k_fold, scoring=scoring)
accuracy2 = result2["test_accuracy"].mean()
precision2 = result2["test_precision"].mean()
recall2 = result2["test_recall"].mean()
f1_score2 = result2["test_f1_score"].mean()
auc_score2 = result2["test_roc_auc_score"].mean()

print("accuracy: {0: .4f}".format(accuracy2))
print("precision: {0: .4f}".format(precision2))
print("recall: {0: .4f}".format(recall2))
print("f1_score: {0: .4f}".format(f1_score2))
print("auc_score: {0: .4f}".format(auc_score2))

#Third 
y3=data['target']
X3=third
k_fold = KFold(n_splits=5, random_state=1, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score': make_scorer(roc_auc_score)
          }

result3 = cross_validate(clf1, X3, y3, cv=k_fold, scoring=scoring)
accuracy3 = result3["test_accuracy"].mean()
precision3 = result3["test_precision"].mean()
recall3 = result3["test_recall"].mean()
f1_score3 = result3["test_f1_score"].mean()
auc_score3 = result3["test_roc_auc_score"].mean()

print("accuracy: {0: .4f}".format(accuracy3))
print("precision: {0: .4f}".format(precision3))
print("recall: {0: .4f}".format(recall3))
print("f1_score: {0: .4f}".format(f1_score3))
print("auc_score: {0: .4f}".format(auc_score3))

#Third 



















