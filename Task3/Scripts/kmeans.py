#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  06kmeans.py
#  

import codecs
import numpy
import sklearn.cluster
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plotdata(data,labels,colors, X_pca, name, type_plot): #def function plotdata
#colors = ['black']
    numbers = numpy.arange(len(X_pca))
    fig, ax = plt.subplots()
    for i in range(len(X_pca)):
        if type_plot == 'scatter':
            plt.scatter(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]])
        if type_plot == 'text':
            plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]])
    plt.xlim(-1, 1.3)
    plt.ylim(-1, 1)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()
    
# 0. Load Data
f = codecs.open("iquitos-train.csv", "r", "utf-8")
dataweeks = []
count = 0
for line in f:
	if count > 0: 
		# remove double quotes
		row = line.replace ('"', '').split(",")
		row.pop(0)
		if row != []:
			dataweeks.append(map(float, row))
	count += 1
	

#1. Preprocessing

# normalization
#1.1. Normalization of the data
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
dataweeks = min_max_scaler.fit_transform(dataweeks)
# dimensionality reduction

	
#2. Setting parameters
k = 2

# 2. Clustering execution

# 2.1 k-means ++ 
distortions = []
silhouettes = []

init = 'k-means++' # initialization method 
iterations = 10 # to run 10 times with different random centroids to choose the final model as the one with the lowest SSE
max_iter = 300 # maximum number of iterations for each single run
tol = 1e-04 # controls the tolerance with regard to the changes in the within-cluster sum-squared-error to declare convergence
random_state = 0

centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(dataweeks, k, init)

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(dataweeks)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(dataweeks, labels))

# Plot distoritions    
plt.plot(range(2, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Plot Silhouette
plt.plot(range(2, 11), silhouettes , marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silohouette')
plt.show()

# 3. Validation

print("Silhouette Coefficient (Kmeans++): %0.3f"
      % metrics.silhouette_score(numpy.asarray(dataweeks), labelsplus))

#4.1 PCA Estimation
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(dataweeks)

print(estimator.explained_variance_ratio_) 

#4.2.  plot
colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)
 
plotdata(dataweeks, labelsplus, colors, X_pca, "Kmeans", "scatter")
plotdata(dataweeks, labelsplus, colors, X_pca, "Kmeans", "text")


