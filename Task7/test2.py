# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:11:16 2017

@author: Edu
"""

import RedPandas as rp
import numpy as np

den_train = rp.loadDataCSV('dengue_train.csv', target='total_cases', 
                null_target_procedure = rp.DELETE_ROW,
                null_procedure = rp.MEAN)

den_test = rp.loadDataCSV('dengue_test.csv', 
                null_target_procedure = rp.DELETE_ROW,
                null_procedure = rp.MEAN)

#Dividing the training into 2 dataframes, Iquitos and San Juan
    
den_train_div = rp.divideDataFrame(den_train, 'city')

for div in den_train_div:
    
    if div.dataFrame['city'].iloc[0] == 'iq':
        iq_train = div
        
    elif div.dataFrame['city'].iloc[0] == 'sj':
        sj_train = div
      
    div.reportBasicInfo()
    
    for report in div.reports:
        report.showReport()

#Dividing the test into 2 dataframes, Iquitos and San Juan
        
den_test_div = rp.divideDataFrame(den_test, 'city')

for div in den_test_div:
    
    if div.dataFrame['city'].iloc[0] == 'iq':
        iq_test = div
        
    elif div.dataFrame['city'].iloc[0] == 'sj':
        sj_test = div
      
    div.reportBasicInfo()
    
    for report in div.reports:
        report.showReport()
    

#Outlier cleaning done in Task5
#Delete outliers 
#1st iteration
iq_train.dataFrame.drop(iq_train.dataFrame.index[[244, 104, 3, 103, 51, 306, 10, 115, 273]],inplace=True)
sj_train.dataFrame.drop(sj_train.dataFrame.index[[507,500,705,800]],inplace=True)
#2nd iteration
iq_train.dataFrame.drop(iq_train.dataFrame.index[[23]],inplace=True)
     
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
for x in range(0,30):
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
for report in iq_train.reports:
    report.showReport()

#Feature selection with the previous tests
iq_train.features = []
iq_train.features.append('weekofyear')
iq_train.features.append('reanalysis_specific_humidity_g_per_kg')
iq_train.features.append('station_avg_temp_c')

print iq_train.features

min_range = 2
max_range = 100
depths = []

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
print depths
#TODO im tired and it's late I will automatize this later
knn_iq_neighbors = 99
knn_iq_weigth = 'uniform'
    
iq_train.setupRegressor(n_neighbors=knn_iq_neighbors,
                        weights='uniform',
                        mode='KNN')

iq_pred = iq_train.regressor.predict(iq_test.dataFrame[iq_train.features])

print iq_pred









