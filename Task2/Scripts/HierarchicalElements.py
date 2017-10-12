# -*- coding: utf-8 -*-

# 1. Load data
import loaddata
data, names = loaddata.load_data("iquitos-train.csv")
import numpy


#1. Data normalizazion
#http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()

rows = 206
cols = 15
print(cols)

datanorm = min_max_scaler.fit_transform(data)

#2. Principal Component Analysis
from sklearn.decomposition import PCA
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(datanorm)

import matplotlib.pyplot as plt
plt.plot(X_pca[:,0], X_pca[:,1],'x')


#3. Hierarchical Clustering
# 3.1. Compute the similarity matrix
import sklearn.neighbors
import numpy
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(X_pca)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3.2. Building the Dendrogram	
from scipy import cluster
clusters = cluster.hierarchy.linkage(matsim, method = 'average')
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
cluster.hierarchy.dendrogram(clusters, color_threshold=6)
plt.show()

cut = 6  # !!!! ad-hoc
labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
print 'Number of clusters %d' % (len(set(labels)))


# 4. plot
colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
plt.xlim(-1, 2)
plt.ylim(-1, 1)
ax.grid(True)
fig.tight_layout()
plt.show()

# 5. characterization
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
     

for c in range(1,n_clusters_+1):
    print 'Group', c
    for i in range(len(datanorm[0])):
        column = [row[i] for j,row in enumerate(data) if labels[j] == c]
        print i, numpy.mean(column)
            
groups=[]
for k in range(numpy.amax(labels)):
    groups.append([])
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
    groups[labels[i]-1].append(numbers[i])

for l in range(len(groups)):
    print "\n\n"
    for m in range(len(groups[l])):
        if((m+1)<len(groups[l])):
            print(groups[l][m], colors[labels[l]],' Distance with next: ' ,groups[l][m+1]-groups[l][m])#,data[m])
        else:
            print(groups[l][m], colors[labels[l]])#,data[m])

        