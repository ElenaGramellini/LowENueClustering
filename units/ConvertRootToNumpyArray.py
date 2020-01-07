import numpy as np
import os
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT


def readTTree(rootFileName = '../rootFiles/Nue_LowE.root', treeName = 'MCNeutrinoAna/anatree'):
    # this is where I read the TTree...
    rfile  = ROOT.TFile(rootFileName)
    intree = rfile.Get(treeName)
    #... and convert the TTree into an array
    array = tree2array(intree)
    # Save away
    outFileName = "../npFiles/" + os.path.basename(rootFileName)[:-4] + "npy"
    print "creating np file as ", outFileName
    np.save(outFileName, array)

readTTree()

