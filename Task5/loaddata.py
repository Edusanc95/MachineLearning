# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:05:59 2016

@author: FranciscoP.Romero
"""
import codecs
def load_data(path):
        
    f = codecs.open(path, "r", "utf-8")
    states = []
    names = []
    count = 0
    for line in f:
        if count > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		#row.pop(0)
		if row != []:
			states.append(map(float, row))
        else:
           names = line.replace ('"', '').split(",")[1:]
        count += 1
 
   
    return states,names