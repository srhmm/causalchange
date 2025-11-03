import numpy as np
from scipy.special import comb

def Combinator(M,k):
	sum=comb(M+k-1, M);
	if sum==0:
		#print(M,",",k," abnormal");
		return 0;
	return np.log2(sum);
    

def CombinatorAccumulate(M,k):
	sum=0;

	for r in range(1,M+1):
		n= k+r-1;
		#print("n: ",n,", r: ",r,"----------> nCr: ",comb(n, r));
		sum = sum + comb(n, r)
        
    #print("Total: ",sum);
	if sum==0:
		return 0;

	return np.log2(sum);    
	
'''
def main():
    
    while(True):
        M = int(input("Max interactions: "));
        k = int(input("Number of parents: "));
        print("Bits: ",combinator(M,k));



main()
'''
