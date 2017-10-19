#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  06kmeans.py
#  

import codecs
import numpy
import sklearn.cluster
from sklearn import metrics

# 0. Load Data
f = codecs.open("iquitos-train.csv", "r", "utf-8")
states = []
count = 0
for line in f:
	if count > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		row.pop(0)
		if row != []:
			states.append(map(float, row))
	count += 1
	

#1. Preprocessing

# normalization
#1.1. Normalization of the data
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
states = min_max_scaler.fit_transform(states)
# dimensionality reduction

	
#2. Setting parameters
k = 4



# 2. Clustering execution

# 2.1 random inicialization
centroids, labels, z =  sklearn.cluster.k_means(states, k, init="random" )

# 2.2 k-means ++ 
centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(states, k, init="k-means++" )


# 3. Validation

print("Silhouette Coefficient (Random): %0.3f"
      % metrics.silhouette_score(numpy.asarray(states), labels))

print("Silhouette Coefficient (Kmeans++): %0.3f"
      % metrics.silhouette_score(numpy.asarray(states), labelsplus))

#4.1 PCA Estimation
from sklearn.decomposition import PCA
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(states)



print(estimator.explained_variance_ratio_) 

#4.2.  plot 
import matplotlib.pyplot as plt
colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
plt.xlim(-1, 1.3)
plt.ylim(-1, 1)
ax.grid(True)
fig.tight_layout()
plt.show()



#print labels
#print labels2