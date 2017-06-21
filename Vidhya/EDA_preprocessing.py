#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:38:59 2017

@author: Catheirne Wang
"""

import pandas as pd 
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

df = pd.read_csv('/Users/zhizhenwang/GoogleDrive/Project/Vidhya/train.csv')

#visualizing some univariate distribution 
df['ApplicantIncome'].hist(bins=50)
df['CoapplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='CoapplicantIncome')

df.boxplot(column='ApplicantIncome',by ='Education')
df.boxplot(column='CoapplicantIncome',by ='Gender')

df['LoanAmount'].hist(bins=50)

#craete pivot table for 
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status', index=['Credit_History'],
                       aggfunc = lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Credit History: ')
print (temp1) 

print('\nProbability of getting loan for each Credit History class:')
print(temp2)


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp4 = pd.crosstab([df['Credit_History'],df['Gender']], df['Loan_Status'])
temp4.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


#imputation of missing value 
df.apply(lambda x: sum(x.isnull()),axis=0)

#fillin the missig values in LoanAmount with mean 
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df.boxplot(column = 'LoanAmount', by=['Education','Self_Employed'])

#impute missing value in Self_Employed 
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)

#create pivot table of mmedian value and replace the missing value using this table
table = df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

def replace_na(x):
    return table.loc[x['Self_Employed'],x['Education']]

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(replace_na, axis=1), inplace=True)

#take care of Gender, Married, Dependents, Loan_Amount_Term, Credit_History
df['Married'].value_counts()
df['Married'].fillna('No',inplace=True)
#df['Dependents'].fillna(0,inplace=True)
#df.ix[df.Dependents==0,'Dependents']='0'

df['Gender'].fillna('Male',inplace=True)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df['Credit_History'])

#normalise extreme value by taking log value 
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)



## Building Predictive Model 'Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'
#,,
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

df.to_csv('/Users/zhizhenwang/GoogleDrive/Project/Vidhya/cleandata.csv')


