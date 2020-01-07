# LowENueClustering
Scope of this project is exploring the possibility of augmenting nue reconstruction efficiency at low energy by discriminating between cosmic rays and low energy nue at the clustering stage.
The core "physics" idea is the following: since low energy nues have uhm uhm... low energy... their products will not travel far in the detector. We expect events to occupy a reasonably small space. This is not true for cosmic rays or high energy neutrino events. 
Can we develop a strategy to keep high efficiency for nues and decent cosmic rejection by rejecting "big" clusters and focus on the smaller ones? Time will tell..


Some techinical details and some sparse thoughts:
1) It is common practice in HEP to use ROOT (https://root.cern.ch/) and root files as data analysis framework and data storage format. 
1.1) The cool kids these days abhor it (for a number of good reasons combined with a fair amount of pretentiousness), mostly in favor of <pick_their_favorite> python package. 
1.2) My position on the whole "the best analysis framework you really need to use" is the same as the whole debate around "the best editor"
