from sklearn import linear_model;
import numpy as np;
from .RFunctions import *

class Slope:
	def __init__(self):
		self.inclusive_model=True;
	
	
	def FitSpline(self,source,target,M=2,temp=False):
		sse,coeff,hinge_count,interactions = REarth(source,target,M)
		score = self.model_score(np.copy(coeff[0])) 
		return sse,score,coeff,hinge_count,interactions;
	
	def FitModel(self,source,target,temp=False):
		reg = linear_model.LinearRegression(fit_intercept =False);
		reg.fit(source,target);
		coeff = reg.coef_;
		sse  = np.sum( (reg.predict(source) - target)**2 );
		score = self.model_score(np.copy(coeff[0]));
		return sse,score,coeff;

	
	def model_score(self,coeff):
		Nans = np.isnan(coeff);
		
		if any(Nans):
			print ('Warning: Found Nans in regression coefficients. Setting them to zero...')
		coeff[Nans]=0;
		sum =0;
		for c in coeff:
			if np.abs(c)>1e-12:
				c_abs =  np.abs(c);
				c_dummy = c_abs;
				precision = 1;
				
				while c_dummy<1000:
					c_dummy *=10;
					precision+=1;
				sum = sum + self.logN(c_dummy) + self.logN(precision) + 1
		return sum;
	
	def logN(self,z):
		z = np.ceil(z);
		
		if z < 1 :
			return 0;
		else :
			log_star = self.logg(z);
			sum = log_star;
			
			while log_star > 0:
				log_star = self.logg(log_star);
				sum = sum+log_star;
			
			return sum + self.logg(2.865064)
		
	def gaussian_score_emp_sse(self,sse, n,min_diff):
		var = sse / n
		sigma = np.sqrt(var)
		return self.gaussian_score_sse(sigma, sse, n,min_diff)

	def gaussian_score_sse (self,sigma, sse, n,resolution):
		sigmasq = sigma**2;
		if sse == 0.0 or sigmasq == 0.0:
			return np.array([0.0]);
		else:
			err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * self.logg(2 * np.pi * sigmasq)) - n * self.logg(resolution)
			return max(err,np.array([0]));
		
	def logg(self,x):
		if x == 0:
			return 0
		else:
			return np.log2(x)
