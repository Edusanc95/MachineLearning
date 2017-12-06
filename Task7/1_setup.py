# -*- coding: utf-8 -*-

import RedPandas as rp

den_train = rp.loadDataCSV('db/dengue_train.csv', target='total_cases', 
                null_target_procedure = rp.DELETE_ROW,
                null_procedure = rp.MEAN)

den_test = rp.loadDataCSV('db/dengue_test.csv', 
                null_target_procedure = rp.DELETE_ROW,
                null_procedure = rp.MEAN)

#Dividing the training into 2 dataframes, Iquitos and San Juan
    
den_train_div = rp.divideDataFrame(den_train, 'city')

for div in den_train_div:
    
    if div.dataFrame['city'].iloc[0] == 'iq':
        iq_train = div
        
    elif div.dataFrame['city'].iloc[0] == 'sj':
        sj_train = div
      
    div.reportBasicInfo(printOnScreen=False)
    

#Dividing the test into 2 dataframes, Iquitos and San Juan
        
den_test_div = rp.divideDataFrame(den_test, 'city')

for div in den_test_div:
    
    if div.dataFrame['city'].iloc[0] == 'iq':
        iq_test = div
        
    elif div.dataFrame['city'].iloc[0] == 'sj':
        sj_test = div
      
    div.reportBasicInfo(printOnScreen=False)
    

#Outlier cleaning done in Task5
#Delete outliers 
#1st iteration
iq_train.dataFrame.drop(iq_train.dataFrame.index[[244, 104, 3, 103, 51, 306, 10, 115, 273]],inplace=True)
sj_train.dataFrame.drop(sj_train.dataFrame.index[[507,500,705,800]],inplace=True)
#2nd iteration
iq_train.dataFrame.drop(iq_train.dataFrame.index[[23]],inplace=True)

iq_train.dataFrame.to_csv('db/iq_train.csv',index=False)
sj_train.dataFrame.to_csv('db/sj_train.csv',index=False)

iq_test.dataFrame.to_csv('db/iq_test.csv',index=False)
sj_test.dataFrame.to_csv('db/sj_test.csv',index=False)