#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:57:30 2017

@author: edu
"""

# 0 load data
import RedPandas as rp
import pandas as pd
import matplotlib.pyplot as plt
weeks, features, n_nulls = rp.loadDataCSV(name='dengue_train.csv',
                              target='total_cases',
                              null_target_procedure=rp.DELETE_ROW,
                              null_procedure=rp.MEAN)

rp.showInfoDF(weeks, features, n_nulls)

#divide into iq and sj
# get a list of city names
cityNames=weeks['city'].unique().tolist()
iqWeeks=weeks.loc[weeks.city=='iq']
sjWeeks=weeks.loc[weeks.city=='sj']

#Outlier cleaning done in Task5
#0 Delete outliers 
#1st iteration
iqWeeks.drop(iqWeeks.index[[244, 104, 3, 103, 51, 306, 10, 115, 273]],inplace=True)
sjWeeks.drop(sjWeeks.index[[507,500,705,800]],inplace=True)
#2nd iteration
iqWeeks.drop(iqWeeks.index[[23]],inplace=True)


#Feature engineering done in Task5
iqFeatures = []
sjFeatures = []
#For iq
iqFeatures.append('weekofyear')
iqFeatures.append('reanalysis_specific_humidity_g_per_kg')
iqFeatures.append('station_avg_temp_c')
iqFeatures.append('reanalysis_dew_point_temp_k')
#For sj
sjFeatures.append('weekofyear')
sjFeatures.append('ndvi_nw')
sjFeatures.append('reanalysis_dew_point_temp_k')
sjFeatures.append('reanalysis_specific_humidity_g_per_kg')
sjFeatures.append('station_min_temp_c')

weeksTest, featuresTest, n_nullsTest = rp.loadDataCSV(name='dengue_test.csv',
                              null_target_procedure=rp.DELETE_ROW,
                              null_procedure=rp.MEAN)

#divide into iq and sj
iqWeeksTest=weeksTest.loc[weeksTest.city=='iq']
sjWeeksTest=weeksTest.loc[weeksTest.city=='sj']
iqWeeksTest.reset_index(inplace=True)
sjWeeksTest.reset_index(inplace=True)

# features and labels
X = iqWeeks[iqFeatures] #Training
y = iqWeeks['total_cases'] #Target
Z = iqWeeksTest[iqFeatures] #Test

# x axis for plotting
import numpy as np
xx = np.stack(i for i in range(len(y)))

# 1. CROSS VALIDATION ANALYSIS
from sklearn import neighbors

for i, weights in enumerate(['uniform', 'distance']):
    rp.crossValidation(iqWeeks, iqFeatures, 'total_cases', mode='KNN',
                       weights=weights)


#We are going to use the distance weight for BOTH

#1. Build the model
n_neighbors = 3 # BEST PARAMETER as in the crossvalidation test
weights='distance'
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

y_pred = knn.fit(X,y).predict(Z)
    
plt.subplot(2, 1, i + 1)
#plt.scatter(xx, y, c='k', label='data')
plt.plot(y_pred, c='r', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

y_pred = y_pred.astype(int)
d = {'city': pd.Series(iqWeeksTest['city']), 'year': pd.Series(iqWeeksTest['year']),
         'weekofyear' : pd.Series(iqWeeksTest['weekofyear']), 'total_cases' : pd.Series(y_pred)}
iqdf = pd.DataFrame(d)
    
plt.show()

# features and labels
X = sjWeeks[sjFeatures]
y = sjWeeks['total_cases']
Z = sjWeeksTest[sjFeatures]

# x axis for plotting
import numpy as np
xx = np.stack(i for i in range(len(y)))

# 1. CROSS VALIDATION ANALYSIS
for i, weights in enumerate(['uniform', 'distance']):
    rp.crossValidation(sjWeeks, sjFeatures, 'total_cases', mode='KNN',
                       weights=weights)
    
#1. Build the model
n_neighbors = 25 # BEST PARAMETER as in the crossvalidation test
weights='distance'
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
y_pred = knn.fit(X,y).predict(Z)
 
plt.subplot(2, 1, i + 1)
#plt.scatter(xx, y, c='k', label='data')
plt.plot(y_pred, c='r', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))
y_pred = y_pred.astype(int)
d = {'city': pd.Series(sjWeeksTest['city']), 'year': pd.Series(sjWeeksTest['year']),
         'weekofyear' : pd.Series(sjWeeksTest['weekofyear']), 'total_cases' : pd.Series(y_pred)}
sjdf = pd.DataFrame(d)
    
plt.show()

result = sjdf.append(iqdf)

result.to_csv('result_dengue.csv')
#delete first column