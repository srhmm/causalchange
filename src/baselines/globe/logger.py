
class Logger:

	def __init__ (self,path,verbose=True,log_to_disk=False):
		self.path=path;
		self.file_handle = "";
		self.verb=verbose;
		self.dummy = not log_to_disk;
	
	def Begin(self):
		if self.dummy:
			return;
		
		self.file_handle=open(self.path,"w+");
	
	def WriteLog(self,str):
		if self.dummy:
			if self.verb:
				print(str);
			return;
			
		self.file_handle.write(str+"\n");
		if self.verb:
			print(str);
			
	def End(self):
		if self.dummy:
			return;
		self.file_handle.close();
		
		
	
		
