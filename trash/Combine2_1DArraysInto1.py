print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from root_numpy import root2array, tree2array
from root_numpy import testdata
import ROOT


tp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print "1", tp.ndim
print tp.shape

fp = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print "2", fp.ndim
print fp.shape

combined = np.vstack((tp, fp)).T
print "combined", combined
print combined.ndim
print combined.shape


