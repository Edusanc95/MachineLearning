# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:11:16 2017

@author: Edu
"""

import RedPandas as rp
import numpy as np
import pandas as pd

iq_train = rp.loadDataCSV('db/iq_train.csv', target='total_cases')

iq_test = rp.loadDataCSV('db/iq_test.csv', target='total_cases')

#Iquitos analysis
iq_train.features.remove('year')
iq_train.features.remove('total_cases')
iq_train.features.remove('city')
iq_train.features.remove('week_start_date')

#CV tests
    #Decision tree
    #This is mostly for feature selection
min_range = 2
max_range = 30
depths = []

#Multiple CV tests done since it's random so we get the most frequent depth
#Range is one so it doesn't take ages to compile, but for the analysis it was at 30
for x in range(0,1):
    iq_tree_score = rp.crossValidation(iq_train, mode='DecisionTree', 
                                       min_range=min_range, max_range=max_range,
                                       printGraph=False)

    max_score_tree_iq = 100
    depth = min_range

    #Best max score
    for score in iq_tree_score:
        if score < max_score_tree_iq:
            max_score_tree_iq = score
            max_depth = depth
        depth = depth + 1
    depths.append([max_depth,max_score_tree_iq])

#We get the depth that appears more
counts = np.bincount([i[0] for i in depths])
#For reference, is almost always 2 for this particular study

print depths
dtree_iq_depth = np.argmax(counts)

#Final depth
print 'Depth decision tree Iquitos '+str(dtree_iq_depth)

#Setting up regressor. 
iq_train.setupRegressor(criterion='mae', 
                        max_depth=dtree_iq_depth,
                        random_state=0, 
                        mode='DecisionTree')

iq_train.reportInfoRelevancies()

#Feature selection with the previous tests

iq_train.features = []
iq_train.features.append('weekofyear')
iq_train.features.append('reanalysis_specific_humidity_g_per_kg')
iq_train.features.append('station_avg_temp_c')
iq_train.features.append('reanalysis_dew_point_temp_k')

print iq_train.features

min_range = 2
max_range = 40
depths = []
'''
    #KNN
for weight in ['uniform','distance']:
    iq_knn_score = rp.crossValidation(iq_train, mode='KNN', weights=weight, 
                   min_range=min_range, max_range=max_range)

    max_score_knn_iq = 100
    depth = min_range

    #Best max score
    for score in iq_knn_score:
        if score < max_score_knn_iq:
            max_score_knn_iq = score
            max_depth = depth
        depth = depth + 1
    depths.append([max_depth,max_score_knn_iq, weight])

#We get the depth that appears more

knn_iq_neighbors = 99
knn_iq_weigth = 'uniform'
    
iq_train.setupRegressor(n_neighbors=knn_iq_neighbors,
                        weights='uniform',
                        mode='KNN')
'''

'''
#More estimators better result, but 4 seems good for the computation time
iq_rf_score = rp.crossValidation(iq_train, mode='RandomForest',n_estimators=4,
                   min_range=min_range, max_range=max_range, criterion='mae')

max_score_rf_iq = 100
depth = min_range

#Best max score
for score in iq_rf_score:
    if score < max_score_rf_iq:
        max_score_rf_iq = score
        max_depth = depth
    depth = depth + 1
depths.append([max_depth,max_score_rf_iq])
#With Knn we get a worse CV score than with random forest, so we roll with that

print depths
'''

df_iq_depth=3
iq_train.setupRegressor(max_depth=df_iq_depth,
                        mode='RandomForest',
                        n_estimators=1000)

#MAE test
rp.computeErrorMeasure(iq_train)
rp.computeMax(iq_train)

iq_pred = iq_train.regressor.predict(iq_test.dataFrame[iq_train.features])

iq_pred = iq_pred.astype(int)
d = {'city': pd.Series(iq_test.dataFrame['city']), 'year': pd.Series(iq_test.dataFrame['year']),
         'weekofyear' : pd.Series(iq_test.dataFrame['weekofyear']), 'total_cases' : pd.Series(iq_pred)}
iqdf = pd.DataFrame(d)

iqdf.to_csv('db/result_iq.csv',index=False)






