import numpy as np;


class Node:

	# input is normalized n-ary variable
	def __init__(self,variable,glb):
		self.globe_=glb;
		self.dims = variable.shape[1];
		self.var = variable;
		self.min_diff=self.CalculateMinDiff(variable);
		self.CalculateDefaultScore(variable);
		
	
	def CalculateDefaultScore(self,variable,):
		value=self.ScoreNode(self,True);
		self.default_score = value;
		self.current_score=value;

	def ScoreNode(self,child,debug=False):
		dt=[];
		rows = child.GetData().shape[0];
		dt.append(child.GetData().reshape(rows,-1)**0);
		
		source =np.hstack(dt);
		target=child.GetData();
		new_bits,coeffs = self.globe_.ComputeScore(source,target,rows,child.GetMinDiff(),k=np.array([1]));
		return new_bits;
	
	def GetCurrentBits(self):
		return self.current_score[0];
	
	def SetCurrentBits(self,new_score):
		self.current_score=max(np.array([1e-5]),new_score);
		
	def GetDefaultScore(self):
		return self.default_score;

	def logg(self,x):
		if x == 0:
			return 0
		else:
			return np.log2(x)
			
	def GetMinDiff(self):
		return self.min_diff;
	
	def GetData(self):
		return self.var;
	
	def GetRowCount(self):
		return self.var.shape[0];

	def CalculateMinDiff(self,variable):
		sorted_v =np.copy(variable);
		sorted_v.sort(axis=0);
		diff = np.abs(sorted_v[1]-sorted_v[0]);
		
		if diff==0: diff=np.array([10.01]);
		
		for i in range(1,len(sorted_v)-1):
			curr_diff=np.abs(sorted_v[i+1]-sorted_v[i]);
			if curr_diff!=0 and curr_diff < diff:
				diff = curr_diff;
		return diff;
	
