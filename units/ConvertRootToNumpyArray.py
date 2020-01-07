import numpy as np
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT


def main(rootFileName = '../rootFiles/Nue_LowE.root', treeName = 'MCNeutrinoAna/anatree'):
    rfile  = ROOT.TFile(rootFileName)
    intree = rfile.Get(treeName)
    # and convert the TTree into an array
    array = tree2array(intree)
    print array

main()

