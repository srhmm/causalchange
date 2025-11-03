import numpy as np

from rpy2.robjects import r
import rpy2.robjects.numpy2ri
from rpy2 import robjects;
from rpy2.robjects.packages import importr
MARS = importr('earth');
SID_ = importr('SID');
PCALG= importr('pcalg');
import re;

rpy2.robjects.numpy2ri.activate()

def REarth(X,Y,M=1):
	row,col=X.shape;
	rX=r.matrix(X,ncol=col,byrow=False);
	rY=r.matrix(Y,ncol=1,byrow=True);
	
	coeffs=[];


	try:
		rearth=MARS.earth(x=rX,y=rY,degree=M);
	except:
		print("Singular fit encountered, retrying with Max Interactions=1");
		rearth=MARS.earth(x=rX,y=rY,degree=1);

	COEFF_INDEX=11;
	CUTS_INDEX=6;
	SELECTED_INDEX=7;
	RSS_INDEX=0;

	arg_count=np.size(rearth[COEFF_INDEX]);
	cut_count=np.size(rearth[SELECTED_INDEX]);

	z=str(rearth[COEFF_INDEX]);
	lines=z.split('\n');
	interactions=[];
	for i in range(2,len(lines)):
		val=1+ lines[i].count('*');		
		interactions.append(val);


	#print(rearth[COEFF_INDEX]);
	#print("-----------");
	#print(rearth[CUTS_INDEX]);
	#print("-----------");
	#print(rearth[SELECTED_INDEX]);
	#print("-----------");
	#'''
	tst=str(rearth[COEFF_INDEX]);
	listed=re.split('\n|h\(x[0-9]{1,1000}\-|h\(|\-x[0-9]{1,1000}\)|\)',tst)	
	#print(listed);
	for vs in listed:
		try:
			if vs is not None:
				coeffs.append(float(vs.strip()));
		except ValueError:
			pass;
	'''
	for i in range(0,arg_count):
		coeffs.append(rearth[COEFF_INDEX][i]);
	#'''

#	print(coeffs);
#	print(len(coeffs));
#	import ipdb; ipdb.set_trace();
	sse=rearth[RSS_INDEX][0]
	#print("sse: ",sse);
	#print(arg_count);
	return sse,[coeffs],np.array([cut_count]),interactions;


def RPC_Gauss(X):
	row,col=X.shape;
	rX=r.matrix(X,ncol=col,byrow=False);
	
	coeffs=[];
	pc=PCALG.PC(x=rX,y=rY) 

	COEFF_INDEX=11;
	RSS_INDEX=0;

	arg_count=np.size(rearth[COEFF_INDEX]);

	for i in range(0,arg_count):
		coeffs.append(rearth[COEFF_INDEX][i]);

	sse=rearth[RSS_INDEX][0]
	
	return sse,[coeffs];


def getStructuralDistances(G,H):
	row,col=G.shape;
	
	rG=r.matrix(G.reshape(-1),ncol=col,byrow=True);
	rH=r.matrix(H.reshape(-1),ncol=col,byrow=True);
	
	SID=SID_.structIntervDist(rG,rH)
	SHD=SID_.hammingDist(rG,rH)
	
	#import ipdb;ipdb.set_trace();
	return SID[2],SID[1],SHD[0]
'''
def main():
	X = np.array([2,1,3,2,4,1,5,1,6,1,7,1,8,2]).reshape(-1,2);
	Y = np.array([8.1,12.78,14.29,17.11,19.5,22.77,27.62]).reshape((-1,1));

	print(X);
	sse,co=REarth(X,Y);
	
	print(sse);
	print(co);
	
main();
'''
