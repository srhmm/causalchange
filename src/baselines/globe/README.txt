----------------------------------------------------------
REQUIREMENTS:
----------------------------------------------------------
- Python version: 3.7 or better
- Packages:
	* scikit-learn
	* numpy
	* mpmath
	* matplotlib
	* cdt (causal discovery toolbox) 

- R Version: 3.6 or better
- R packages:
	* SID: 	 https://www.rdocumentation.org/packages/SID/versions/1.0
	* pcalg: https://www.rdocumentation.org/packages/pcalg/versions/2.6-8
	* kpcalg/RCIT:  https://github.com/Diviyan-Kalainathan/RCIT
	* spresebn: https://www.rdocumentation.org/packages/sparsebn/versions/0.1.0
	* bnlearn: https://www.rdocumentation.org/packages/bnlearn/versions/4.5


##################################################
#  GLOBE - Implementation of our work titled	 # 
#  "Discovering Fully Oriented Causal Networks"  #
#  in Proceedings of AAAI-2021.					 #
##################################################
	
- This version of GLOBE has been cleaned and commented for the ease of use. The implementation can be further optimized but is not in my immediete plans.
- If you would like to reproduce the results of the experiments stated in the publication, you can find the data and the (automated) version of GLOBE on this link: https://www.dropbox.com/sh/iuy4cv7uzn54m6u/AADH5C7wdC-jQG73RD4rZOXAa?dl=0
- There are two files that mainly map to the "Algorithm" section of the publication: SkeletonHandler.sh (Contains the edge ranking phase) and DAG.py (contains Forward and Backward Search phases)
- For scoring functions mentioned in the published work, refer to globe.py
- This code is not optimized for performance and may contain some unused variables within. Although, I have done my best to resolve the latter.	


Author: Osman Ali Mian
Last Modified: 22nd August 2022
Contact: osman.mian@cispa.de
