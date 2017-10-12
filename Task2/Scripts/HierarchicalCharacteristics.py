import matplotlib.pyplot as plt
import numpy


#http://docs.scipy.org/doc/scipy/reference/cluster.html
from scipy import cluster
from sklearn import preprocessing 
import sklearn.neighbors

# 0. Load Data
import loaddata
states,names = loaddata.load_data("iquitos-train_with_name.csv")
features = numpy.transpose(states);

#1. Normalization of the data
#http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)


#1.2. Principal Component Analysis
from sklearn.decomposition import PCA
estimator = PCA (n_components = 3)
X_pca = estimator.fit_transform(features_norm)
print("Variance Ratio: ", estimator.explained_variance_ratio_) 


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
print(len(names))
for i in range(len(X_pca)):
    print(i)
    plt.text(X_pca[i][0], X_pca[i][1], names[i]) 


plt.xlim(-2, 6)
plt.ylim(-1, 1.5)
ax.grid(True)
fig.tight_layout()
plt.show()





	
# 2. Compute the similarity matrix
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(features_norm)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3. Building the Dendrogram	
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
clusters = cluster.hierarchy.linkage(matsim, method = 'single')
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
cluster.hierarchy.dendrogram(clusters, color_threshold = 3, labels = names, leaf_rotation=90)
plt.show()