import numpy as np
import os
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT
import LoadNumpyArray as ldA

def checkLoadedTree(rootFileName = '../rootFiles/Nue_LowE.root', treeName = 'MCNeutrinoAna/anatree'):
    # this is where I read the TTree...
    rfile  = ROOT.TFile(rootFileName)
    intree = rfile.Get(treeName)
    #... and convert the TTree into an array
    array        = tree2array(intree)
    loaded_array = ldA.load()
    print array == loaded_array

checkLoadedTree()

