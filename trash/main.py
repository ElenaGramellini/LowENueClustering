import numpy as np
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT
from ROOT import TCanvas, TH2D, TH1D
import ClusterAlgo as ca
from ClusterInfo import ClusterInfo as ci


def findBiggestCluster(clusters):
    if not len(clusters):
        return 0
    maxHits = 0
    biggestIt = 0
    for i in xrange(len(clusters)):
        if maxHits < clusters[i].nHits:
            maxHits = clusters[i].nHits
            biggestIt = i
    #print clusters[biggestIt].nHits
    return clusters[biggestIt]


    
def main():
    #rfile = ROOT.TFile("Nue_LowE.root")
    rfile = ROOT.TFile("Nue_Overlay.root")
    intree = rfile.Get('MCNeutrinoAna/anatree')
    outFile = ROOT.TFile("OutNue_Overlay.root","recreate")
    hClusterWireWidth_All = TH1D("hClusterWireWidth_All","hClusterWireWidth_All",500,0,500)
    hClusterWireWidth_Big = TH1D("hClusterWireWidth_Big","hClusterWireWidth_Big",500,0,500)
    # convert the TTree into an array
    array = tree2array(intree)
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
        clusters0 = ca.computeClustersOnPlane(wire0,time0)
        clusters1 = ca.computeClustersOnPlane(wire1,time1)
        clusters2 = ca.computeClustersOnPlane(wire2,time2)

        meanTime0 = []
        meanTime1 = []
        meanTime2 = []
        for c in xrange(len(clusters0)):
            thisCluster = clusters0[c]
            meanTime0.append(int(thisCluster.avgTime))
            hClusterWireWidth_All.Fill(thisCluster.wireWidth())

        for c in xrange(len(clusters1)):
            thisCluster = clusters1[c]
            meanTime1.append(int(thisCluster.avgTime))
            hClusterWireWidth_All.Fill(thisCluster.wireWidth())

        for c in xrange(len(clusters2)):
            thisCluster = clusters2[c]
            meanTime2.append(int(thisCluster.avgTime))
            hClusterWireWidth_All.Fill(thisCluster.wireWidth())

        print sorted(meanTime0)
        print sorted(meanTime1)
        print sorted(meanTime2)
        #print meanTime0
        #print meanTime1
        #print meanTime2
        bigC0 = findBiggestCluster(clusters0)
        bigC1 = findBiggestCluster(clusters1)
        bigC2 = findBiggestCluster(clusters2)
        hClusterWireWidth_Big.Fill(bigC0.wireWidth())
        hClusterWireWidth_Big.Fill(bigC1.wireWidth())
        hClusterWireWidth_Big.Fill(bigC2.wireWidth())
        #raw_input()

    #outFile.Add(hClusterWireWidth_All)
    outFile.Write()
    outFile.Close()

main()
