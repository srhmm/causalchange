import mpmath;

class Scorer:

	def score_edge(self,Gain,Sigma):
		
		G=Gain[0];
		S=Sigma[0];
		
		if G<=0:
			return mpmath.ln(1);

		if G<S:
			S=G;
		
		eps = mpmath.ln( 1.0/mpmath.ln(2)) / mpmath.ln(2);
		
		t0 = mpmath.power(2,-G-eps);
		
		t1 = mpmath.power(2,-S-eps);
		#print 't1: ',t1;
		t2 = (1.0/G) * mpmath.power(2,-G);
		#print 't2: ',t2;
		t3 = mpmath.power(t1,1) + mpmath.power(t2,1);
		#print 't3: ',t3;
		t4 = -(S)#mpmath.ln(mpmath.power(2,-S));
		#t4 = mpmath.ln(t0 * t3);
		
		return t4;
		
	def score_edge_crude(self,Gain,Sigma):
		G=Gain[0];
		S=Sigma[0];
		if G<=0:
			return mpmath.ln(1);
			
		t0 = mpmath.power(2,-(G+S));
		t1 = mpmath.ln(t0);
		return t1;