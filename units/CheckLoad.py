import numpy as np
from root_numpy import root2array, tree2array
import ROOT
import LoadNumpyArray as ldA

import os
project_home = os.environ['LOW_NUE_CLUSTER_HOME'] 

def checkLoadedTree(rootFileName =  project_home+'/rootFiles/Nue_LowE.root', treeName = 'MCNeutrinoAna/anatree'):
    # this is where I read the TTree...
    rfile  = ROOT.TFile(rootFileName)
    intree = rfile.Get(treeName)
    #... and convert the TTree into an array
    array        = tree2array(intree)
    #... let's load the array using the load function, which reads the already stored file
    loaded_array = ldA.load(project_home+'/npFiles/Nue_LowE.npy')
    # if the loaded array is the same as the original one, you should see only "True" printed. 
    print array == loaded_array

checkLoadedTree()

