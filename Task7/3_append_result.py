# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:11:16 2017

@author: Edu
"""
import pandas as pd

sjdf = pd.read_csv('db/result_sj.csv')
iqdf = pd.read_csv('db/result_iq.csv')

result = sjdf.append(iqdf)
result.to_csv('db/result_dengue.csv',index=False)
#After this I still have to modify the csv by hand so I have to check it