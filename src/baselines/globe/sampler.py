import numpy as np;
from copy import deepcopy;

class Sampler:
	def __init__(self):
		self.initiated=True

		
	def Mutate(self,v1):
		epsilon= np.mean(v1)/10.0;
		rows = v1.shape[0];
		
		eps_arr = np.random.uniform(low=-epsilon, high=epsilon, size=rows).reshape((rows,-1));
		mutated_array = v1 + eps_arr;
		np.random.shuffle(mutated_array)
		
		return mutated_array
			
	

