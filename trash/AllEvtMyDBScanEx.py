print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT
from ROOT import TCanvas, TH2D

## Eventually, we'll want a "cluster info class"
## Right now, clusterInfo is a list containing [nHists, min wire, max wire, min time, max time]


def clusterDecision(allClustersInfo):
    tempNHits = 0
    theKey    = 0
    if len(allClustersInfo) == 0:
        return []
    for k in allClustersInfo.keys():
        if allClustersInfo[k][0] > tempNHits:
            tempNHits = allClustersInfo[k][0]
            theKey = k
    return allClustersInfo[theKey]


def findWidestClusterInWire(allClustersInfo):
    tempDist = 0
    timeDiff = 0
    theKey   = 0
    if len(allClustersInfo) == 0:
        return []
    for k in allClustersInfo.keys():
        thisDistW = allClustersInfo[k][2] - allClustersInfo[k][1]  
        if thisDistW > tempDist:
            tempDist = thisDistW
            timeDiff = allClustersInfo[k][4] - allClustersInfo[k][3]
            theKey = k
    return allClustersInfo[theKey], tempDist, timeDiff


def findWidestClusterInTime(allClustersInfo):
    tempDist = 0
    theKey   = 0
    if len(allClustersInfo) == 0:
        return []
    for k in allClustersInfo.keys():
        thisDistT = allClustersInfo[k][4] - allClustersInfo[k][3]  
        if thisDistT > tempDist:
            tempDist = thisDistT
            theKey = k
    return allClustersInfo[theKey], tempDist



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


def Plot(X0,X1,X2,run,subrun,evt,drawMe=True):
    db0 = DBSCAN(eps=0.3, min_samples=5).fit(X0)
    core_samples_mask0 = np.zeros_like(db0.labels_, dtype=bool)
    core_samples_mask0[db0.core_sample_indices_] = True
    L0 = db0.labels_

    db1 = DBSCAN(eps=0.3, min_samples=5).fit(X1)
    core_samples_mask1 = np.zeros_like(db1.labels_, dtype=bool)
    core_samples_mask1[db1.core_sample_indices_] = True
    L1 = db1.labels_

    db2 = DBSCAN(eps=0.3, min_samples=5).fit(X2)
    core_samples_mask2 = np.zeros_like(db2.labels_, dtype=bool)
    core_samples_mask2[db2.core_sample_indices_] = True
    L2 = db2.labels_
    
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_0 = len(set(L0)) - (1 if -1 in L0 else 0)
    n_noise_0 = list(L0).count(-1)
    n_clusters_1 = len(set(L1)) - (1 if -1 in L1 else 0)
    n_noise_1 = list(L1).count(-1)
    n_clusters_2 = len(set(L2)) - (1 if -1 in L2 else 0)
    n_noise_2 = list(L2).count(-1)

    #print 'Estimated number of clusters: (', n_clusters_0, n_clusters_1, n_clusters_2,")"
    #print 'Estimated number of noise points: (', n_noise_0, n_noise_1, n_noise_2,")"

    if not drawMe:
        return L0, L1, L2

    import matplotlib.pyplot as plt
    plt.figure(1,facecolor='white',figsize=(10,10))
    stupidLabel = str(run) + " "+ str(subrun) + " "+str(evt)

    if  1:
        # #############################################################################
        # Plot result
        plt.subplot(311)
        # Black removed and is used for noise instead.
        unique_L0 = set(L0)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_L0))]
        for k, col in zip(unique_L0, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (L0 == k)
                
            xy = X0[class_member_mask & core_samples_mask0]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
    
            xy = X0[class_member_mask & ~core_samples_mask0]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        
        plt.title(stupidLabel+' Estimated number of clusters: %d' % n_clusters_0)

    # Number of clusters in labels, ignoring noise if present.
    if  1:
        # Plot result
        plt.subplot(312)
        # Black removed and is used for noise instead.
        unique_L1 = set(L1)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_L1))]
        for k, col in zip(unique_L1, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (L1 == k)
                
            xy = X1[class_member_mask & core_samples_mask1]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
    
            xy = X1[class_member_mask & ~core_samples_mask1]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        
        plt.title('Estimated number of clusters: %d' % n_clusters_1)


    # Number of clusters in labels, ignoring noise if present.
    if 1:
        # #############################################################################
        # Plot result
        plt.subplot(313)
        # Black removed and is used for noise instead.
        unique_L2 = set(L2)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_L2))]
        for k, col in zip(unique_L2, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (L2 == k)
                
            xy = X2[class_member_mask & core_samples_mask2]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
    
            xy = X2[class_member_mask & ~core_samples_mask2]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        
        plt.title('Estimated number of clusters: %d' % n_clusters_2)
        plt.show()
        return L0, L1, L2

# #############################################################################
rfile = ROOT.TFile("Nue_LowE.root")
#rfile = ROOT.TFile("Nue_Overlay.root")
intree = rfile.Get('MCNeutrinoAna/anatree')

# and convert the TTree into an array
array = tree2array(intree)
hScatter = TH2D("hScatter","hScatter",14,0.,0.7,150,0,900)
hScatterCm = TH2D("hScatterCm","hScatter",14,0.,0.7,300,0,270)
hScatterTime = TH2D("hScatterTime","hScatter",14,0.,0.7,300,0,1200)
hScatterCm1 = TH2D("hScatterCm1","hScatter",14,0.,0.7,300,0,270)
for i in xrange(0,len(array)):
    print i
    run   = array[i][ 0]
    subrun= array[i][ 1]
    evt   = array[i][ 2]
    trueE = array[i][ 6]
    wire0 = array[i][10]
    time0 = array[i][13]
    wire1 = array[i][11]
    time1 = array[i][14]
    wire2 = array[i][12]
    time2 = array[i][15]
    X0 = np.vstack((wire0, time0)).T
    X1 = np.vstack((wire1, time1)).T
    X2 = np.vstack((wire2, time2)).T
    if not len(X0):
        continue
    if not len(X1):
        continue
    if not len(X2):
        continue
    #X0 = StandardScaler().fit_transform(X0)
    #X1 = StandardScaler().fit_transform(X1)
    #X2 = StandardScaler().fit_transform(X2)
    L0, L1, L2 = Plot(X0,X1,X2,run,subrun,evt)

    clusters0 = readLabels(wire0,time0,L0) 
    clusters1 = readLabels(wire1,time1,L1) 
    clusters2 = readLabels(wire2,time2,L2) 

    c_Ind1 = (findWidestClusterInWire(clusters0))
    c_Ind2 = (findWidestClusterInWire(clusters1))
    c_Coll = (findWidestClusterInWire(clusters2))

    #ct_Ind1 = (findWidestClusterInTime(clusters0))
    #ct_Ind2 = (findWidestClusterInTime(clusters1))
    #ct_Coll = (findWidestClusterInTime(clusters2))


    i1 = 0
    i2 = 0
    c  = 0
    if len(c_Ind1):
        i1 = c_Ind1[1]
        hScatter.Fill(trueE, c_Ind1[1])
        hScatterCm.Fill(trueE, 0.3*c_Ind1[1])
        hScatterTime.Fill(trueE, c_Ind1[2])
    if len(c_Ind2):   
        i2 = c_Ind2[1]                
        hScatter.Fill(trueE, c_Ind2[1])
        hScatterCm.Fill(trueE, 0.3*c_Ind2[1])
        hScatterTime.Fill(trueE, c_Ind2[2])
    if len(c_Coll):
        c = c_Coll[1]
        hScatter.Fill(trueE, c_Coll[1])
        hScatterCm.Fill(trueE, 0.3*c_Coll[1])
        hScatterTime.Fill(trueE, c_Coll[2])
    m = max(i1,i2,c)
    hScatterCm1.Fill(trueE, 0.3*m)


hScatter.Draw("colz")
raw_input()
