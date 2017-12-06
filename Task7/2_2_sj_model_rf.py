# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 16:00:23 2017

@author: Edu
"""

import RedPandas as rp
import numpy as np
import pandas as pd

sj_train = rp.loadDataCSV('db/sj_train.csv', target='total_cases')

sj_test = rp.loadDataCSV('db/sj_test.csv', target='total_cases')

#San Juan analysis
sj_train.features.remove('year')
sj_train.features.remove('total_cases')
sj_train.features.remove('city')
sj_train.features.remove('week_start_date')

#CV tests
    #Decision tree
    #This is mostly for feature selection
min_range = 2
max_range = 30
depths = []

#Multiple CV tests done since it's random so we get the most frequent depth

for x in range(0,1):
    sj_tree_score = rp.crossValidation(sj_train, mode='DecisionTree', 
                                       min_range=min_range, max_range=max_range,
                                       printGraph=False)

    max_score_tree_sj = 100
    depth = min_range

    #Best max score
    for score in sj_tree_score:
        if score < max_score_tree_sj:
            max_score_tree_sj = score
            max_depth = depth
        depth = depth + 1
    depths.append([max_depth,max_score_tree_sj])

#We get the depth that appears more
counts = np.bincount([i[0] for i in depths])
#For reference, is almost always 2 for this particular study

print depths
dtree_sj_depth = np.argmax(counts)

#Final depth
print 'Depth decision tree san juan '+str(dtree_sj_depth)

#Setting up regressor. 
sj_train.setupRegressor(criterion='mae', 
                        max_depth=dtree_sj_depth,
                        random_state=0, 
                        mode='DecisionTree')

sj_train.reportInfoRelevancies()

#Feature selection with the previous tests
#TODO automatization
sj_train.features = []
sj_train.features.append('weekofyear')
sj_train.features.append('ndvi_nw')
sj_train.features.append('reanalysis_dew_point_temp_k')
sj_train.features.append('reanalysis_specific_humidity_g_per_kg')
#sj_train.features.append('station_min_temp_c')

print sj_train.features

min_range = 2
max_range = 40
depths = []

'''
    #KNN
for weight in ['uniform','distance']:
    sj_knn_score = rp.crossValidation(sj_train, mode='KNN', weights=weight, 
                   min_range=min_range, max_range=max_range)

    max_score_knn_sj = 100
    depth = min_range

    #Best max score
    for score in sj_knn_score:
        if score < max_score_knn_sj:
            max_score_knn_sj = score
            max_depth = depth
        depth = depth + 1
    depths.append([max_depth,max_score_knn_sj, weight])

#We get the depth that appears more
print depths

knn_sj_neighbors = 22
knn_sj_weigth = 'uniform'
    
sj_train.setupRegressor(n_neighbors=knn_sj_neighbors,
                        weights='uniform',
                        mode='KNN')
'''
'''
#More estimators better result, but 4 seems good for the computation time
sj_rf_score = rp.crossValidation(sj_train, mode='RandomForest',n_estimators=4,
                   min_range=min_range, max_range=max_range, criterion='mae')

max_score_rf_sj = 100
depth = min_range

#Best max score
for score in sj_rf_score:
    if score < max_score_rf_sj:
        max_score_rf_sj = score
        max_depth = depth
    depth = depth + 1
depths.append([max_depth,max_score_rf_sj])
#With Knn we get a worse CV score than with random forest, so we roll with that

print depths
'''

'''
#Create a Gaussian Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
# Parametrization
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
model = BernoulliNB()
# training the model
model.fit(sj_train.dataFrame[sj_train.features],sj_train.dataFrame[sj_train.target])
# prediction with the same data
sj_pred = model.predict(sj_test.dataFrame[sj_train.features])
'''

df_sj_depth=10
sj_train.setupRegressor(max_depth=df_sj_depth,
                        mode='RandomForest',
                        n_estimators=1000)

#MAE test
rp.computeErrorMeasure(sj_train)
rp.computeMax(sj_train)

sj_pred = sj_train.regressor.predict(sj_test.dataFrame[sj_train.features])

sj_pred = sj_pred.astype(int)
d = {'city': pd.Series(sj_test.dataFrame['city']), 'year': pd.Series(sj_test.dataFrame['year']),
         'weekofyear' : pd.Series(sj_test.dataFrame['weekofyear']), 'total_cases' : pd.Series(sj_pred)}

sjdf = pd.DataFrame(d)

sjdf.to_csv('db/result_sj.csv',index=False)