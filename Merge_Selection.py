#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:34:32 2020

@author: kimjong-gyu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('/Users/kimjong-gyu/Desktop/airbnb-recruiting-new-user-bookings/df.csv')
df_session=pd.read_csv('/Users/kimjong-gyu/Desktop/airbnb-recruiting-new-user-bookings/session_pre.csv')

df_session.isnull().sum()

df_session=df_session.drop(['Unnamed: 0'],axis=1)

df=df.drop(['Unnamed: 0'],axis=1)

df_session.rename(columns={"user_id":"id"},inplace=True)
df_session.columns

merge_left = pd.merge(df_session,df, how='left', left_on='id', right_on='id')


merge_left.head(50)