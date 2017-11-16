# -*- coding: utf-8 -*-

# 1. Load data
import RedPandas as RP

data, name, n_nulls = RP.loadDataCSV(name='dengue_train.csv',
                              target='total_cases',
                              null_target_procedure=RP.DELETE_ROW,
                              null_procedure=RP.MEAN)
name = list(data.head(0))
import numpy

#divide into iq and sj
'''
# set the index to be this and don't drop
data.set_index(keys=['city'], drop=False,inplace=True)
'''
# get a list of city names
cityNames=data['city'].unique().tolist()
iqWeeks=data.loc[data.city=='iq']
sjWeeks=data.loc[data.city=='sj']

name.remove('total_cases')
name.remove('city')
name.remove('year')
name.remove('week_start_date')

#0 Delete outliers 
#1st iteration
iqWeeks.drop(iqWeeks.index[[244, 104, 3, 103, 51, 306, 10, 115, 273]],inplace=True)
sjWeeks.drop(sjWeeks.index[[507,500,705,800]],inplace=True)
#2nd iteration
iqWeeks.drop(iqWeeks.index[[23]],inplace=True)
#1. Data normalizazion
#http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()

datanorm = min_max_scaler.fit_transform(sjWeeks[name])

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

cut = 14  # !!!! ad-hoc
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
plt.ylim(-1, 1.5)
ax.grid(True)
fig.tight_layout()
plt.show()

# 5. characterization
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

        