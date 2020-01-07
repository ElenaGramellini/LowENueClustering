import numpy as np
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT
from ROOT import TCanvas, TH2D, TH1D
import ClusterAlgo as ca
from sklearn.preprocessing import StandardScaler
from ClusterInfo import ClusterInfo as ci
import copy


def findBiggestCluster(clusters):
    if not len(clusters):
        return 0
    maxHits = 0
    biggestIt = 0
    for i in xrange(len(clusters)):
        if maxHits < clusters[i].nHits:
            maxHits = clusters[i].nHits
            biggestIt = i
    return clusters[biggestIt]


def scaleDefinition():
    rfile  = ROOT.TFile("Nue_LowE.root")
    intree = rfile.Get('MCNeutrinoAna/anatree')
    array  = tree2array(intree)
    wire0  = array[0][10]
    time0  = array[0][13]
    X0     = np.vstack((wire0, time0)).T
    scaler = StandardScaler().fit(X0)
    return scaler


def main():
    #compute the scaling from test set
    scaler = scaleDefinition()

    # Read cosmic file
    rfile = ROOT.TFile("Nue_Overlay.root")
    intree = rfile.Get('MCNeutrinoAna/anatree')
    array  = tree2array(intree)
    hWide     = TH1D("Wide","hScatter",300,0,2500)
    #hScatter     = TH2D("hScatter","hScatter",14,0.,0.7,150,0,900)
    #hScatterCm   = TH2D("hScatterCm","hScatter",14,0.,0.7,300,0,270)
    #hScatterTime = TH2D("hScatterTime","hScatter",14,0.,0.7,300,0,1200)
    for i in xrange(0,len(array)):
        print i
        run   = array[i][ 0]
        subrun= array[i][ 1]
        evt   = array[i][ 2]
        trueE = array[i][ 6]
        wire0 = array[i][10]
        time0 = array[i][13]
        # make the 2D data set
        X_0     = np.vstack((wire0, time0)).T
        # scale the test data set to previously calculated mean and std
        X_0     = scaler.transform(X_0)
        clusterOutPut   = ca.Cluster(X_0)
        allClustersInfo = ca.readLabels(wire0,time0,clusterOutPut[1]) 
        #ca.Draw(clusterOutPut[0],X_0,clusterOutPut[1])
        #print allClustersInfo
        
        for aci in allClustersInfo.values():
            hWide.Fill(aci.wireWidth())
    outFile = ROOT.TFile("OutNue_Overlay.root","recreate")    
    outFile.Add(hWide)
    outFile.Write()
    outFile.Close()

main()
