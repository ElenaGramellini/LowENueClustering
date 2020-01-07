import numpy as np


def load(npFileName = '../npFiles/Nue_LowE.npy'):
    array = np.load(npFileName,allow_pickle = True )
    return array

load()

