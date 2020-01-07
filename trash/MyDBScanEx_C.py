print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT


## Eventually, we'll want a "cluster info class"
## Right now, clusterInfo is a list containing [nHists, min wire, max wire, min time, max time]

def readLabels(w, t, l):
    # let's determine the important cluster information: min/max wires and times
    allClustersInfo    = {} # key = cluster label, value = this clusterInfo
    allClustersInfoAll = {} # key = cluster label, value = [vector wires, vector times]

    # loop over the labels (= they determine if the hits belong to the same cluster)
    for i in xrange(len(l)):
        # ignore noise
        if l[i] < 0: 
            continue
        # if the key already exists, modify the value already present
        # by adding a wire and a time in the wire and time lists
        if l[i] in allClustersInfoAll.keys():
            oldPair = allClustersInfoAll[ l[i] ]
            oldPair[0].append(w[i])
            oldPair[1].append(t[i])
        # if the key does NOT already exists
        # create a new value object (= a pair of wire and time arrays)
        else:
            wires = [w[i]]
            times = [t[i]]
            pair  = [wires, times]
            allClustersInfoAll[ l[i] ] = pair

    # looping on the clusters, let's keep only the important info
    for k in  allClustersInfoAll.keys():
        minW = min(allClustersInfoAll[k][0])
        maxW = max(allClustersInfoAll[k][0])
        minT = min(allClustersInfoAll[k][1])
        maxT = max(allClustersInfoAll[k][1])
        clusterInfo = [len(allClustersInfoAll[k][0]), minW,maxW,minT,maxT]
        allClustersInfo[k] = clusterInfo

    return allClustersInfo


# #############################################################################
# Get 1 event
rfile = ROOT.TFile("Nue_LowE.root")
intree = rfile.Get('MCNeutrinoAna/pot_tree')

# and convert the TTree into an array
array = tree2array(intree)
wire0 = array[0][ 9]
time0 = array[0][12]
wire1 = array[0][10]
time1 = array[0][13]
wire2 = array[0][11]
time2 = array[0][14]

X = np.vstack((wire2, time2)).T
X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print "labels size:",len(labels), len(X), labels

infoClusters = readLabels(wire2, time2, labels)
for k in infoClusters.keys():
    print k, infoClusters[k]

#print "db", db

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
