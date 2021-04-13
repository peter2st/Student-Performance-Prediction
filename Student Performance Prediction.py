# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:51:20 2021

@author: peter2st
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('C:/Users/scott/Desktop/Programming Stuff/Data_sets/StudentsPerformance.csv')

df.rename(columns={"test preparation course": "prep", "math score": "math", "reading score": "reading", "writing score": "writing", 'parental level of education': 'parental', 'race/ethnicity': 'race'}, inplace=True)

#plotting gender
sns.set_style('whitegrid')
sns.countplot(y='gender',data=df,palette='winter')
plt.show()

#plotting race
sns.countplot(y='race', data=df, palette='winter')
plt.show()

#plotting whether or not the student completed the test prep
sns.countplot(y='prep', data=df, palette='winter')
plt.show()

#plotting lunch
sns.countplot(y='lunch', data=df, palette='winter')
plt.show()

#plotting the parental level of education
sns.countplot(y='parental', data=df, palette='winter')
plt.show()

df['gender'].replace('female', 1,inplace=True)
df['gender'].replace('male', 0,inplace=True)

df['lunch'].replace('standard', 1,inplace=True)
df['lunch'].replace('free/reduced', 0,inplace=True)

df['prep'].replace('completed', 1,inplace=True)
df['prep'].replace('none', 0,inplace=True)

print(df.race.unique())

df['groupA'] = df['race'].apply(lambda x: 1 if x == 'group A' else 0)
df['groupB'] = df['race'].apply(lambda x: 1 if x == 'group B' else 0)
df['groupC'] = df['race'].apply(lambda x: 1 if x == 'group C' else 0)
df['groupD'] = df['race'].apply(lambda x: 1 if x == 'group D' else 0)
df['groupE'] = df['race'].apply(lambda x: 1 if x == 'group E' else 0)
df = df.drop(['race'], 1)

df['parental'] = df['parental'].apply(lambda x: 0 if x == 'some high school' else
                                      (1 if x == 'high school' else
                                       (2 if x == 'some college' else
                                        (3 if x == 'associate\'s degree' else
                                         (4 if x == 'bachelor\'s degree' else 5)))))

max_math = max(df['math'])
min_math = min(df['math'])
df['math'] = df['math'].apply(lambda x: ((x - min_math)/(max_math - min_math)))

max_reading = max(df['reading'])
min_reading = min(df['reading'])
df['reading'] = df['reading'].apply(lambda x: ((x - min_reading)/(max_reading - min_reading)))

max_writing = max(df['writing'])
min_writing = min(df['writing'])
df['writing'] = df['writing'].apply(lambda x: ((x - min_writing)/(max_writing - min_writing)))
df['avg'] = (df['math'] + df['reading'] + df['writing']) / 3

df['pass'] = df['avg'].apply(lambda x: 1 if x >= .60 else 0)

pd.set_option('display.max_columns', None)
print(df.head())
lg = LinearRegression()

#features = ['gender', 'lunch', 'prep', 'groupA', 'groupB', 'groupC', 'groupD', 'groupE', 'parental']
features = ['gender', 'lunch', 'prep']
print(df.head())

# #Predict math scores
# x = df[features]
# y = df[['math']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
# lg.fit(x_train, y_train)
# #math_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 5]]) * 100
# math_predict = lg.predict([[1, 1, 0]]) * 100
# math_predict = round(math_predict[0][0], 2)
# print('According to your information, I predict you will get a',math_predict,'% on your math test.')

# #Predict reading scores
# x = df[features]
# y = df[['reading']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
# lg.fit(x_train, y_train)
# #reading_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 5]]) * 100
# reading_predict = lg.predict([[1, 1, 0]]) * 100
# reading_predict = round(reading_predict[0][0], 2)
# print('According to your information, I predict you will get a',reading_predict,'% on your reading test.')

# #Predict writing scores
# x = df[features]
# y = df[['writing']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
# lg.fit(x_train, y_train)
# #writing_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 5]]) * 100
# writing_predict = lg.predict([[1, 1, 0]]) * 100
# writing_predict = round(writing_predict[0][0], 2)
# print('According to your information, I predict you will get a',writing_predict,'% on your writing test.')

# #Predict average score
# x = df[features]
# y = df[['avg']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
# lg.fit(x_train, y_train)
# #avg_predict = lg.predict([[1, 1, 0, 0, 1, 0, 0, 0, 5]]) * 100
# avg_predict = lg.predict([[1, 1, 0]]) * 100
# avg_predict = round(avg_predict[0][0], 2)
# print('According to your information, I predict you will get a',avg_predict,'% average.')

#Predict pass or fail
log = LogisticRegression()
x = df[features]
y = df[['pass']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)
log.fit(x_train, y_train.values.ravel())

#Predict chance of pass or fail
#pass_predict = log.predict([[1, 1, 0, 0, 1, 0, 0, 0, 5]])
#percent_pass = log.predict_proba([[1, 1, 0, 0, 1, 0, 0, 0, 5]])
# pass_predict = log.predict([[1, 1, 0]])
# percent_pass = log.predict_proba([[1, 1, 0]])
# pass_perc = round(percent_pass[0][1], 3) * 100
# fail_perc = round(percent_pass[0][0], 3) * 100
# print('According to your information, I predict that there is a',pass_perc,'% chance you will pass, and a',fail_perc,'% chance you will fail.')
# print(pass_predict)
#Let's plot some graphs to display some of the stats of the students performances

pass_count = 0
fail_count = 0
for i in df['pass']:
    if i == 1:
        pass_count += 1
    else:
        fail_count += 1

plt.bar('Pass', pass_count)
plt.bar('Fail', fail_count)
plt.show()

sns.countplot(y='pass', data=df, palette='winter')
plt.show()

lg.fit(x_train, y_train)

print(lg.score(x_test, y_test))
print(log.score(x_test, y_test))


x_train, x_test, y_train, y_test = train_test_split(x, np.ravel(y), train_size = 0.8, test_size = 0.2, random_state = 100)
forest = RandomForestClassifier(n_estimators = 2)
forest.fit(x_train, y_train)
score = forest.score(x_test, y_test)
print(score)
print(lg.coef_)
print(log.coef_)

print(forest.predict([[0, 0, 1]]))
