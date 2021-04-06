# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:51:20 2021

@author: peter2st
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics

df = pd.read_csv('C:/Users/scott/Desktop/Programming Stuff/Data_sets/StudentsPerformance.csv')

df.rename(columns={"test preparation course": "prep", "math score": "math", "reading score": "reading", "writing score": "writing"}, inplace=True)
df['avg'] = df.mean([['math', 'reading', 'writing']])


df['gender'].replace('female', 1,inplace=True)
df['gender'].replace('male', 0,inplace=True)

df['lunch'].replace('standard', 1,inplace=True)
df['lunch'].replace('free/reduced', 0,inplace=True)

df['prep'].replace('completed', 1,inplace=True)
df['prep'].replace('none', 0,inplace=True)

df['groupA'] = df['race/ethnicity'].apply(lambda x: 1 if x == 'group A' else 0)
df['groupB'] = df['race/ethnicity'].apply(lambda x: 1 if x == 'group B' else 0)
df['groupC'] = df['race/ethnicity'].apply(lambda x: 1 if x == 'group C' else 0)
df['groupD'] = df['race/ethnicity'].apply(lambda x: 1 if x == 'group D' else 0)
df['groupE'] = df['race/ethnicity'].apply(lambda x: 1 if x == 'group E' else 0)
df = df.drop(['race/ethnicity'], 1)

df['bachelors'] = df['parental level of education'].apply(lambda x: 1 if x == 'bachelor\'s degree' else 0)
df['masters'] = df['parental level of education'].apply(lambda x: 1 if x == 'master\'s degree' else 0)
df['associates'] = df['parental level of education'].apply(lambda x: 1 if x == 'associate\'s degree' else 0)
df['somecollege'] = df['parental level of education'].apply(lambda x: 1 if x == 'some college' else 0)
df['highschool'] = df['parental level of education'].apply(lambda x: 1 if x == 'high school' else 0)
df['somehigh'] = df['parental level of education'].apply(lambda x: 1 if x == 'some high school' else 0)
df = df.drop(['parental level of education'], 1)

max_math = max(df['math'])
min_math = min(df['math'])
df['math'] = df['math'].apply(lambda x: ((x - min_math)/(max_math - min_math)))

max_reading = max(df['reading'])
min_reading = min(df['reading'])
df['reading'] = df['reading'].apply(lambda x: ((x - min_reading)/(max_reading - min_reading)))

max_writing = max(df['writing'])
min_writing = min(df['writing'])
df['writing'] = df['writing'].apply(lambda x: ((x - min_writing)/(max_writing - min_writing)))

pd.set_option('display.max_columns', None)
print(df.head())
lg = LinearRegression()
         
#Predict math scores
x = df[['gender', 'lunch', 'prep', 'groupA', 'groupB', 'groupC', 'groupD', 'groupE', 'bachelors', 'masters', 'associates', 'somecollege', 'highschool', 'somehigh']]
y = df[['math']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
lg.fit(x_train, y_train)

math_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) * 100
print('According to your information, I predict you will get a',math_predict,'% on your math test.')

#Predict reading scores
x = df[['gender', 'lunch', 'prep', 'groupA', 'groupB', 'groupC', 'groupD', 'groupE', 'bachelors', 'masters', 'associates', 'somecollege', 'highschool', 'somehigh']]
y = df[['reading']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
lg.fit(x_train, y_train)
reading_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) * 100
print('According to your information, I predict you will get a',reading_predict,'% on your reading test.')

#Predict writing scores
x = df[['gender', 'lunch', 'prep', 'groupA', 'groupB', 'groupC', 'groupD', 'groupE', 'bachelors', 'masters', 'associates', 'somecollege', 'highschool', 'somehigh']]
y = df[['writing']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
lg.fit(x_train, y_train)
writing_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) * 100
print('According to your information, I predict you will get a',writing_predict,'% on your writing test.')



log = LogisticRegression()






















