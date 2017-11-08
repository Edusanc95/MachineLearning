# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 20:42:55 2017

@author: Edu
"""

def correlationFeaturesTarget(dataFrame, features, target):
    # NOTE: Low correlation means there's no linear relationship; 
    # it doesn't mean there's no information in the feature that predicts the target.
    corr = []

    for feature in features:
        local_corr = pearsonr(weeks[feature], weeks['total_cases'])[0]
        corr.append(local_corr)

    y_pos = np.arange(len(features))
 
    plt.bar(y_pos, corr, align='center', alpha=0.5)
    plt.xticks(y_pos, features, rotation='vertical')
    plt.ylabel('Correlation')
    plt.title('Correlation features vs target')

    plt.show()

def computeMax(dataFrame, features, target):
    # Compute the max 
    mae = []
    for i in range(2, 30):
        regressor = DecisionTreeRegressor(max_depth=i)
        regressor.fit(dataFrame[features], dataFrame[target])
        pred_values = regressor.predict(dataFrame[features])
        maev = mean_absolute_error(dataFrame[target],pred_values)
        mae.append(maev)

    # Plot mae   
    plt.plot(range(2,30), mae, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('mae')
    plt.show()

def crossValidation(dataFrame, features, target):
    total_scores = []

    for i in range(2, 30):
        regressor = DecisionTreeRegressor(max_depth=i)
        regressor.fit(dataFrame[features], dataFrame[target])
        scores = -cross_val_score(regressor, dataFrame[features],
            dataFrame[target], scoring='neg_mean_absolute_error', cv=10)
        total_scores.append(scores.mean())

    plt.plot(range(2,30), total_scores, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('cv score')
    plt.show()    

def relevantFeaturesCrossValidation(dataFrame, features, relevancies, target):

    plt.figure(figsize=(8,6))
    for relevancy in relevancies:
        #relevancy[0] is the name of the feature
        #relevancy[1] is the relevancy of the feature
        if relevancy[1] > 0:
            #10 is the usual number of iterations for the cv
            #Here we calculate how long is going to be the prediction based on the max and minimum values of the feature
            xx = np.array([np.linspace(np.amin(dataFrame[relevancy[0]])*0.95, 
                                       np.amax(dataFrame[relevancy[0]])*1.05, 
                                       10)]).T
            
            regressor.fit(dataFrame[relevancy[0]].reshape(-1, 1), dataFrame['total_cases'])
            plt.plot(dataFrame[relevancy[0]].reshape(-1, 1), dataFrame[target], 'o', label='observation')
            plt.plot(xx, regressor.predict(xx), linewidth=4, alpha=.7, label='prediction')
            plt.xlabel(relevancy[0])
            plt.ylabel(target)
            plt.legend()
            plt.show()
    
    
# load data
import pandas as pd
weeks = pd.read_csv('iquitos-train.csv')

features = list(weeks.head(0))

#We delete names of the features that are identifiers and te total cases
features.remove('total_cases')
features.remove('city')
features.remove('year')
features.remove('weekofyear')
features.remove('week_start_date')

print features

#Explore Data
# charts from https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
import matplotlib.pyplot as plt
plt.rcdefaults()

#Univariate Histograms
#
weeks.hist()
plt.show()

import numpy as np
from scipy.stats.stats import pearsonr

correlationFeaturesTarget(weeks, features, 'total_cases')

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

computeMax(weeks, features, 'total_cases');

# CROSS VALIDATION ANALYSIS
from sklearn.cross_validation import cross_val_score

crossValidation(weeks, features, 'total_cases')

# Model Parametrization 
#we have to use mae since it's the one used for the competition
regressor = DecisionTreeRegressor(criterion='mae', max_depth=2, random_state=0)
regressor.fit(weeks[features[0]].reshape(-1, 1), weeks['total_cases'])

# Model construction

print 'Feature Relevancies'
regressor.fit(weeks[features], weeks['total_cases'])
list1 = zip(features, regressor.feature_importances_)

#Install tabulate conda install -c conda-forge tabulate
from tabulate import tabulate
print tabulate(list1)

#Model Visualization
import graphviz 
# Don't forget pip install graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(regressor, out_file='tree.dot', feature_names=features, 
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)

from subprocess import check_call
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])

relevantFeaturesCrossValidation(weeks, features, list1, 'total_cases')