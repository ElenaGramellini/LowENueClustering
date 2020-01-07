import numpy as np
import os
project_home = os.environ['LOW_NUE_CLUSTER_HOME'] 

def load(npFileName = project_home+'/npFiles/Nue_LowE.npy'):
    array = np.load(npFileName,allow_pickle = True )
    return array

load()

