# -*- coding: utf-8 -*-
"""
Created on Thu May 13 05:47:28 2021

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("AMC.csv")

df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d')

# plt.plot(df['Close'])


data = df[['Date', 'Close']].copy()


data['Year'] = ""
data['Month'] = ""
data['Week'] = ""
data['Day'] = ""
data['7DAYS'] = ""
data['5DAYS'] = ""

# Create features
for i in range(len(data)):
    data.at[i, 'Year'] = data['Date'][i].year
    data.at[i, 'Month'] = data['Date'][i].month
    data.at[i, 'Week'] = data['Date'][i].week
    data.at[i, 'Day'] = data['Date'][i].day
    if i >= 7:
        seven_avg = 0
        for j in range(1, 8):
            seven_avg += data.at[i-j, 'Close']
        data.at[i, '7DAYS'] = seven_avg/7
    else:
        data.at[i, '7DAYS'] = data.at[i, 'Close']
    if i >= 5:
        five_avg = 0
        for j in range(1, 6):
            five_avg += data.at[i-j, 'Close']
        data.at[i, '5DAYS'] = five_avg/5
    else:
        data.at[i, '5DAYS'] = data.at[i, 'Close']
        
# X_train, X_test, y_train, y_test = train_test_split(data.drop(['Close', 'Date'], axis = 1), data['Close'])

x_train = data.drop(['Close', 'Date'], axis = 1)[:1000]
y_train = data['Close'][:1000]

x_test = data.drop(['Close', 'Date', '7DAYS', '5DAYS'], axis = 1)[1000:]
y_test = data['Close'][1000:]

x_test['7DAYS'] = ""
x_test['5DAYS'] = ""

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

past_closes = data['Close'][len(x_train)-8:len(x_train)].to_list()
for i in range(len(x_test)):
    seven_avg = 0
    five_avg = 0
    for j in range(0, 7):
        seven_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '7DAYS'] = seven_avg/7
    for j in range(0, 5):
        five_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '5DAYS'] = five_avg/5
    test = x_test.iloc[[i]]
    past_closes.append(linear_model.predict(x_test.iloc[[i]]))

plt.plot(df['Close'])

x_test['Predictions'] = past_closes[8:]

plt.plot(x_test['Predictions'])
plt.show()

x_test.drop(['Predictions'], axis = 1, inplace = True)
# KNN
# rmse = []
# for i in range(1, 100):
#     knn_model = KNeighborsRegressor(n_neighbors = i)
#     knn_model.fit(x_train, y_train)
#     knn_preds = knn_model.predict(x_train)
#     rmse.append(np.sqrt(np.mean(np.power((np.array(y_train)-np.array(knn_preds)),2))))

    
knn_model = KNeighborsRegressor(n_neighbors = 1)
knn_model.fit(x_train, y_train)

past_closes = data['Close'][len(x_train)-8:len(x_train)].to_list()
for i in range(len(x_test)):
    seven_avg = 0
    five_avg = 0
    for j in range(0, 7):
        seven_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '7DAYS'] = seven_avg/7
    for j in range(0, 5):
        five_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '5DAYS'] = five_avg/5
    test = x_test.iloc[[i]]
    past_closes.append(knn_model.predict(x_test.iloc[[i]]))

plt.plot(df['Close'])
x_test['Predictions'] = past_closes[8:]

plt.plot(x_test['Predictions'])
plt.show()
x_test.drop(['Predictions'], axis = 1, inplace = True)


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100)
# Train the model on training data
rf.fit(x_train, y_train);

past_closes = data['Close'][len(x_train)-8:len(x_train)].to_list()
for i in range(len(x_test)):
    seven_avg = 0
    five_avg = 0
    for j in range(0, 7):
        seven_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '7DAYS'] = seven_avg/7
    for j in range(0, 5):
        five_avg += past_closes[len(past_closes)-1-j]
    x_test.at[i+1000, '5DAYS'] = five_avg/5
    test = x_test.iloc[[i]]
    past_closes.append(rf.predict(x_test.iloc[[i]]))
plt.plot(df['Close'])

x_test['Predictions'] = past_closes[8:]

plt.plot(x_test['Predictions'])
plt.show()
x_test.drop(['Predictions'], axis = 1, inplace = True)
