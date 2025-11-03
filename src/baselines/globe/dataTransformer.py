import numpy as np;

class DataTransformer:

	def __init__(self,inclusive):
		self.inc=inclusive;
		
	def TransformData(self,x,function_id,offset=0):
		x+=offset;
		#function_id=0;
		if self.inc:
			return self.TransformDataInclusive(x,function_id);
		else:
			return self.TransformDataExclusive(x,function_id);
			
	def TransformDataExclusive(self,s,function_id):
		
		zero_indices = np.where(s == 0)[0];
		eps = 0.0001;
		#dt = np.dtype('Float64')
		x = np.zeros((s.shape[0],1))#,dtype=dt);
			
		if function_id == 0:
			x[:,0] = s.reshape(-1)**1;
		elif function_id == 1:
			x[:,0] = s.reshape(-1)**2;
		elif function_id == 2:
			x[:,0] = s.reshape(-1)**3;
		elif function_id == 3:
			x[:,0] = np.exp(s).reshape(-1);
		elif function_id == 4:
			x[:,0] = s.reshape(-1)**1;
			x[zero_indices,0] = eps;
			x[:,0] = 1.0/x[:,0] ;
		elif function_id == 5:
			x[:,0] = s.reshape(-1)**2;
			x[zero_indices,0] = eps;
			x[:,0] = 1.0/x[:,0] ;
		elif function_id == 6:
			x[:,0] = s.reshape(-1)**4;
		elif function_id==7:
			x[:,0] = np.abs(s).reshape(-1)**0.5;
		else:
			print ('WARNING: unknown function id',function_id, 'encountered, returning column as-is...');
			x[:,0] = x.reshape(-1)**1;
			
		return x;
		
	def TransformDataInclusive(self,s,function_id):
		zero_indices = np.where(s == 0)[0];
		negative_indices = np.where(s < 0)[0];
		eps = 0.0001;
		#dt = np.dtype('Float64');
		if function_id == 0:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
		elif function_id == 1:
			x = np.zeros((s.shape[0],2))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
			x[:,1] = s.reshape(-1)**2;
		elif function_id == 2:
			x = np.zeros((s.shape[0],3))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
			x[:,1] = s.reshape(-1)**2;
			x[:,2] = s.reshape(-1)**3;
		elif function_id == 3:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = np.exp(s).reshape(-1);
		elif function_id == 4:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
			x[zero_indices,0] = eps;
			x[:,0] = 1.0/x[:,0] ;
		elif function_id == 5:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = s.reshape(-1)**2;
			x[zero_indices,0] = eps;
			x[:,0] = 1.0/x[:,0] ;
		elif function_id == 6:
			x = np.zeros((s.shape[0],4))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
			x[:,1] = s.reshape(-1)**2;
			x[:,2] = s.reshape(-1)**3;
			x[:,3] = s.reshape(-1)**4;
		elif function_id==7:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = np.abs(s).reshape(-1)**0.5;
		elif function_id==8:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = np.abs(s).reshape(-1);
			x[zero_indices,0] = eps;
			x[:,0] = 1.0/x[:,0]**0.5 ;
		else:
			print ('WARNING: unknown function id',function_id, 'encountered, returning column as-is...');
			x = np.zeros((s.shape[0],1),dtype=dt);
			import ipdb; ipdb.set_trace();
			x[:,0] = x.reshape(-1)**1;
			
		return x;
