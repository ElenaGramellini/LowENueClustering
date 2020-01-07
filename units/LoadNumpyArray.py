import numpy as np
import os
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT


def load(npFileName = '../npFiles/Nue_LowE.npy'):
    array = np.load(npFileName,allow_pickle = True )
    return array

load()

