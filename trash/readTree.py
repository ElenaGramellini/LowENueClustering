#filename = testdata.get_filepath('Nue_LowE.root')
#arr = root2array(filename, 'MCNeutrinoAna/pot_tree')

from root_numpy import root2array, tree2array
from root_numpy import testdata

# Or first get the TTree from the ROOT file
import ROOT
rfile = ROOT.TFile("Nue_LowE.root")
intree = rfile.Get('MCNeutrinoAna/pot_tree')

# and convert the TTree into an array
array = tree2array(intree)
print array.ndim 
print array.shape 
print array.size 
print array.dtype
print array.itemsize
print 
print array[0][14], len(array[0][14])


raw_input()
