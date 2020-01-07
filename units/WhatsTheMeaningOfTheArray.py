import numpy as np
import LoadNumpyArray as ldA
import os
project_home = os.environ['LOW_NUE_CLUSTER_HOME']

def main():
    print "The unit convention followed for this array is the following:"
    print " - Energy in GeV"
    print " - Distances in cm"
    print " - Time in uB time ticks"
    print " - Plane numbering from cathode-outward... which means"
    print "     0: induction plane closest to the cathode"
    print "     1: induction plane in the middle of the wire planes"
    print "     2: collection plane"
    print " - Wires in uB wire order, which roughly corresponds to:"
    print "     Plane 0: [   0 - 2399]"
    print "     Plane 1: [2400 - 4799]"
    print "     Plane 2: [4800 - 8255]"
    print
    array = ldA.load(project_home + '/npFiles/Nue_LowE.npy')
    print "Array format: "
    print array.dtype

main()

