import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from ClusterInfo import ClusterInfo as ci
from statistics import mean
import matplotlib.pyplot as plt

# this function is the one that actually does the clustering
# it takes the wire-time matrix, it scales it and does the clustering
# the output is a series of labels: each hit belonging to a given cluster has the same label
def ScaleAndCluster(X0):
    if not len(X0):
        return []
    X0 = StandardScaler().fit_transform(X0)
    db0 = DBSCAN(eps=0.3, min_samples=5).fit(X0)
    core_samples_mask0 = np.zeros_like(db0.labels_, dtype=bool)
    core_samples_mask0[db0.core_sample_indices_] = True
    return db0.labels_


def Cluster(X0):
    if not len(X0):
        return []
    db0 = DBSCAN(eps=0.3, min_samples=5).fit(X0)
    core_samples_mask0 = np.zeros_like(db0.labels_, dtype=bool)
    core_samples_mask0[db0.core_sample_indices_] = True
    return db0.labels_ , core_samples_mask0


def Draw(labels, X0, core_samples_mask0):
    plt.figure(1,facecolor='white',figsize=(10,10))
    unique_labels = set(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
                
        xy = X0[class_member_mask & core_samples_mask0]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
    
        xy = X0[class_member_mask & ~core_samples_mask0]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

# this function reads the wire, time and lable object and computes 
# the relevant information for each cluster
# the output is a list of "clusterInfo"
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
        avgW = mean(allClustersInfoAll[k][0])
        minT = min(allClustersInfoAll[k][1])
        maxT = max(allClustersInfoAll[k][1])
        avgT = mean(allClustersInfoAll[k][1])
        clusterInfo = ci(len(allClustersInfoAll[k][0]), minW,maxW,avgW, minT,maxT, avgT)
        #print clusterInfo.nHits, clusterInfo.maxWire, clusterInfo.minWire, clusterInfo.wireWidth()
        allClustersInfo[k] = clusterInfo

    if not len(allClustersInfo):
        allClustersInfo[0] =  ci(0, -1, -2, -1, -1, -2, -1)
    return allClustersInfo


# Generic calculator of clusters per plane
def computeClustersOnPlane(wire, time):
    X  = np.vstack((wire, time)).T
    L  = Cluster(X)
    all_clusters_info = readLabels(wire,time,L)
    return all_clusters_info
