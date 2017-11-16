#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:12:35 2017

@author: edu
"""

import RedPandas as RP

weeks, features, n_nulls = RP.loadDataCSV(name='dengue_train.csv',
                              target='total_cases',
                              null_target_procedure=RP.DELETE_ROW,
                              null_procedure=RP.MEAN)

#divide into iq and sj
# set the index to be this and don't drop
#weeks.set_index(keys=['city'], drop=False,inplace=True)
# get a list of city names
cityNames=weeks['city'].unique().tolist()
iqWeeks=weeks.loc[weeks.city=='iq']
sjWeeks=weeks.loc[weeks.city=='sj']

#0 Delete outliers 
#1st iteration
iqWeeks.drop(iqWeeks.index[[244, 104, 3, 103, 51, 306, 10, 115, 273]],inplace=True)
sjWeeks.drop(sjWeeks.index[[507,500,705,800]],inplace=True)
#2nd iteration
iqWeeks.drop(iqWeeks.index[[23]],inplace=True)

#RP.showInfoDF(iqWeeks, features, n_nulls)
#RP.showInfoDF(sjWeeks, features, n_nulls)

#We delete names of the features that are identifiers and the total cases
features.remove('total_cases')
features.remove('city')
#features.remove('year')
#features.remove('weekofyear')
features.remove('week_start_date')

#Explore Data
# charts from https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
import matplotlib.pyplot as plt
plt.rcdefaults()
#Univariate Histograms
weeks.hist()
plt.show()

#Correlation between features and target feature
RP.correlationFeaturesTarget(iqWeeks, features, 'total_cases')
RP.correlationFeaturesTarget(sjWeeks, features, 'total_cases')

#Check overfitting
#RP.computeMax(weeks, features, 'total_cases');

# CROSS VALIDATION ANALYSIS
RP.crossValidation(iqWeeks, features, 'total_cases')
RP.crossValidation(sjWeeks, features, 'total_cases')

# Model Parametrization and construction
#we have to use mae since it's the one used for the competition
iqRegressor = RP.setupDecisionTreeRegressor(dataFrame=iqWeeks, features=features,
                                          target='total_cases', 
                                          criterion='mae', max_depth=2, 
                                          random_state=0)

sjRegressor = RP.setupDecisionTreeRegressor(dataFrame=sjWeeks, features=features,
                                          target='total_cases', 
                                          criterion='mae', max_depth=3, 
                                          random_state=0)

RP.showInfoRelevancies(features, iqRegressor)
RP.showInfoRelevancies(features, sjRegressor)

#Model Visualization
RP.visualizeTree(iqRegressor, features, 'iqTree.dot')
RP.visualizeTree(sjRegressor, features, 'sjTree.dot')

#Check how the selected features fit the data
#RP.relevantFeaturesCrossValidation(weeks, features, iqRegressor, 'total_cases')
#RP.relevantFeaturesCrossValidation(weeks, features, sjRegressor, 'total_cases')