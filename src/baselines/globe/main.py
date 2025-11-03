from .globeWrapper import GlobeWrapper;
import sys;

def main():
	filename= "data/experiment1.txt"
	Max_Interactions=2;	#See the Instantiation section of the publication
	log_results=True;	#Set this to true if you would like to store the log of the experiment to a text file
	verbose=True;	#Set this to true if you would like see the log output printed to the screen

	globe= GlobeWrapper(Max_Interactions,log_results,verbose);
	
	globe.loadData(filename);
	network = globe.run();

	print(network);

main();


	
