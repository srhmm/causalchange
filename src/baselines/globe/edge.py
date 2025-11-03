import numpy as np;

class Edge:
	
	def __init__ (self,f_id,coefficients,sb,scr, present=False,absent=False):
		self.score=0;
		self.function_id =f_id;
		self.coefficients = coefficients;
		self.saved_bits=sb;
		self.score=scr;
		self.isPresent=present;
		self.isAbsent=absent;
		
	def GetFunctionId(self):
		return self.function_id;
		
	def GetCoefficients(self):
		return self.coefficients;
		
	def GetSavedBits(self):
		return self.saved_bits;
	
	def GetScore(self):
		return self.score;