# LowENueClustering
Scope of this project is exploring the possibility of augmenting nue reconstruction efficiency at low energy by discriminating between cosmic rays and low energy nue at the clustering stage.
The core "physics" idea is the following: since low energy nues have uhm uhm... low energy... their products will not travel far in the detector. We expect events to occupy a reasonably small space. This is not true for cosmic rays or high energy neutrino events. 
Can we develop a strategy to keep high efficiency for nues and decent cosmic rejection by rejecting "big" clusters and focus on the smaller ones? Time will tell..


Some techinical details and some sparse thoughts:
-1. This code is in python 2.7 ... and that needs to change. I have not gotten around and updated my system.
0. Once this is downloaded in the normal github fashion, just source the setup environment (I assume you've got a bash shell)
```
source simpleSetup.sh
```
this will setup the enviromental enviroment LOW_NUE_CLUSTER_HOME to the current directory. This variable is our point of reference for the rest of the code. Most of the rest of the code which is currently there works just by
```
python <name_of_the_script.py>
```

1. It is common practice in HEP to use ROOT (https://root.cern.ch/) and root files as data analysis framework and data storage format. The cool kids these days abhor it (for a number of good reasons combined with a fair amount of pretentiousness), mostly in favor of <pick_their_favorite> python package. I think ROOT is not going to be the future of data analysis in the widest possible sense... while it's quite clear that everyone and their cousins will write a python application one day or another. Yet, ROOT is a very important tool in our field (for a number of good reasons combined with a fair amount of nostalgia)... so it's a good idea to know its ropes.  I'd say my position is: know enough ROOT to get by, explore what works best for your application, don't waste time in arguing on what's "the best analysis framework" (cause that's as fun and as productive as discussing "what's the best editor" or poking each others with sticks) . So here we are: an old caryatid (me, trained mostly in ROOT) trying to use new toys (numpy & sklearn) to find some neutrinos.
2. Our data is in a ROOT format called TTree. So we convert it to numpy arrays using a package called root_numpy. 
A simple example on how to do that is in >  units/npArrayCode/ConvertRootToNumpyArray.py 
I'm following this indication on how to save/load numpy arrays: https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly/55058828
3. We're going to try to cluster hist using scikit-learn, cause I trust computer scientists more than physicists to do that. https://scikit-learn.org/stable/modules/clustering.html#dbscan
