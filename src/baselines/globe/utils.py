import matplotlib;
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np;
import os;
from heapq import heappush, heappop
import operator as op
from functools import reduce
from .dataCleaner import DataCleaner
from copy import deepcopy
from .RFunctions import getStructuralDistances



def WriteOutput(filename,G,k):
	mat = np.array(G);
	mat[mat !=0] =1;
	mat=mat.T
	#import ipdb;ipdb.set_trace();
	with open(filename,'w') as file:
		for i in range(0,k):
			for j in range(0,k-1):
				val=mat[i,j];
				file.write(str(val)+",");
			val=mat[i,k-1]
			file.write(str(val)+"\n");


def CalculatePrecRecallUD(truth,hyp):
	
	r,c= np.shape(truth);
	t = np.copy(truth);
	h = np.copy(hyp);
	
	for i in range(r):
		for j in range(i+1,r):
			if t[i,j] != t[j,i]:
				t[i,j]=1;
				t[j,i]=1;
				
	for i in range(r):
		for j in range(i+1,r):
			if h[i,j] != h[j,i]:
				h[i,j]=1;
				h[j,i]=1;
				
	#import ipdb; ipdb.set_trace();
			
	return CalculatePrecRecall(t,h);

def CalculatePrecRecall(truth,hyp):
	tp_fp = np.sum(hyp);
	tp_fn = np.sum(truth);
	tp = np.count_nonzero(np.logical_and(truth,hyp));
	
	prec=0;
	rec=0;
	F1=0;
	if tp >0:
		prec = tp/tp_fp;
		rec = tp/tp_fn;
		F1 = 2 * (prec*rec) / (prec+rec);
	
	
	
	return [prec,rec,F1]
	

def GTAdjacency(headers,G,H,transpose=False):
	zh= np.array(H,dtype=float);
	zh=np.ceil(abs(zh));
	zh[zh!=0]=1;
	zh=np.array(zh,dtype=int);
	dims=len(G);
	#print(zh)
	
	if transpose:
		zh=zh.T;
		
	zg=zh*0;
	
	for pairs in G:
		print(pairs);
		
		p1=pairs[0];
		p2=pairs[1];
		
		i1=headers[p1];
		i2=headers[p2];
		
		zg[i1,i2]=1;
	
	return zg,zh;



def PrintStructuralStatsNT(zg,zh):
	sid_l,sid_u,shd= getStructuralDistances(zg,zh);
	return int(sid_l[0]),int(sid_u[0]),int(shd);

#G is ground truth
#H is hypothesis
def PrintStructuralStats(headers,G,H):
	zg,zh=GTAdjacency(headers,G,H,True);
	sid_l,sid_u,shd= getStructuralDistances(zg,zh);
	prec,rec,F1 = CalculatePrecRecall(zg,zh);
	uprec,urec,uF1 = CalculatePrecRecallUD(zg,zh);
	return int(sid_l[0]),int(sid_u[0]),int(shd),prec,rec,F1,uprec,urec,uF1;

	
def GetEmptyGraph(dims):
	G=[];
	for a in range(dims):
		G.append([]);
		for b in range(dims):
			G[a].append([]);
	return deepcopy(G);

def PrintAverageStats(auprcd,auprcu,sidul,siduu,shdu,fprec,frec,F1,uprec,urec,uF1,folder,param=True):
	avgd= np.mean(np.array(auprcd));
	avgu= np.mean(np.array(auprcu));
	avgsidl=np.mean(np.array(sidul));
	avgsidu=np.mean(np.array(siduu));
	avgshd=np.mean(np.array(shdu));
	
	if param:
		opt="_poly";
	else:
		opt="_spline";
	
	with open("./tresults/uavgd_auprc"+opt+".txt", "a") as myfile:
		myfile.write(str(folder)+": "+str(avgu)+"\n")
	with open("./tresults/davgd_auprc"+opt+".txt", "a") as myfile:
		myfile.write(str(folder)+": "+str(avgd)+"\n")
		
	with open("./tresults/sid"+opt+".txt", "a") as myfile:
		myfile.write(str(folder)+": ["+str(avgsidl)+","+str(avgsidu)+"]\n")
		
	with open("./tresults/shd"+opt+".txt", "a") as myfile:
		myfile.write(str(folder)+": "+str(avgshd)+"\n")
		
	#with open("./tresults/F1"+opt+".txt", "a") as myfile:
	#	myfile.write(str(folder)+": "+str(fprec)+","+str(frec)+","+str(F1)+"\n")
		
	with open("./tresults/"+str(folder)+"/summary"+opt+".txt", "w") as myfile:
		print("Writing summary for structure");
		print("./tresults/"+str(folder)+"/summary"+opt+".txt")
		for i in range(len(shdu)):
			myfile.write(str(sidul[i])+","+str(siduu[i])+","+str(shdu[i])+"\n");
		myfile.write("=============\n");
		
	with open("./tresults/"+str(folder)+"/roc_summary"+opt+".txt", "w") as myfile:
		print("Writing summary for roc");
		print("./tresults/"+str(folder)+"/roc_summary"+opt+".txt")
		for i in range(len(shdu)):
			myfile.write(str(fprec[i])+","+str(frec[i])+","+str(F1[i])+","+str(uprec[i])+","+str(urec[i])+","+str(uF1[i])+"\n");
		myfile.write("=============\n");

def PrintTimeStats(times,folder):
	minomino=np.mean(np.array(times),axis=0)
	minomino2=np.std(np.array(times),axis=0)
	with open("D:/averages_syn.txt", "a") as myfile:
		myfile.write(str(folder)+": "+"\n");
		myfile.write(str(minomino[0])+","+str(minomino[1])+","+str(minomino[2])+"\n")
		myfile.write("=============\n");
		
	with open("D:/sdevs_syn.txt", "a") as myfile:
		myfile.write(str(folder)+": "+"\n");
		myfile.write(str(minomino2[0])+","+str(minomino2[1])+","+str(minomino2[2])+"\n")
		myfile.write("=============\n");

def PrintGroundTruth(Nodes,gt,headers,path):
	for pairs in gt:
		first = pairs[0];
		second = pairs[1];
		#import ipdb; ipdb.set_trace();
		#find the index in headers where value = first
		i = headers[first];  #[k for k, v in headers.iteritems() if v == first][0]
		j = headers[second]; #[k for k, v in headers.iteritems() if v == second][0]
		fname="Truth_"+first+"_"+second+".png";
		
		source,target = DataCleaner().Clean(Nodes[:,i],Nodes[:,j],2);
		#print(source);
		#print(target);
		Plot2d(source,target,path+fname,True);
		
def PrintAllPairs(Nodes,dims,path):
	if not os.path.exists(path+"all/"):
		os.makedirs(path+"all/");
		
	for i in range(dims):
		for j in range(i+1,dims):
			fname="Visualization_"+str(i)+"_"+str(j)+".png";
			source = Nodes[:,i]
			target = Nodes[:,j];
			#source,target = DataCleaner.Clean(Nodes[:,i],Nodes[:,j]);
			Plot2d(source,target,path+"all/"+fname,True);

def LoadData9(filename):
	with open(filename,'r') as file:
		k = file.readlines();
	
	start=0#1;
	dims = 11#int(k[0].strip());
	recs = len(k)#-1;
	dt = np.dtype('Float64')
	variables= np.zeros((1,dims),dtype=dt);
	
	
	for i in range(start,recs+start):
		if 'nan' not in k[i].lower():
			line = k[i].split(',');
			temp=np.zeros((1,dims),dtype=dt);
			for j in range(0,dims):
				temp[0,j]=line[j].strip();
				
			variables=np.vstack((variables,temp));
		else:
			#print('ignoring...');
			recs=recs-1;
	variables=np.delete(variables,0,0);
	
	return dims,variables,recs;
		
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom
#**************************DATA LOADING******************************#
def LoadData(filename):
	with open(filename,'r') as file:
		k = file.readlines();
	
	start=1;
	dims = int(k[0].strip());
	recs = len(k)-1;
	#dt = np.dtype('Float64')
	variables= np.zeros((1,dims));
	
	
	for i in range(start,recs+start):
		if 'nan' not in k[i].lower():
			line = k[i].split(',');
			temp=np.zeros((1,dims),dtype=dt);
			for j in range(0,dims):
				temp[0,j]=line[j].strip();
				
			variables=np.vstack((variables,temp));
		else:
			#print('ignoring...');
			recs=recs-1;
	variables=np.delete(variables,0,0);
	
	return dims,variables,recs;
	
#******************DATA MANIPULATION FUNCTIONS****************************#
def Store(vars,id,dc):
	nvars = np.around(vars,decimals=3);
	np.savetxt("./tdata/spre_"+str(dc)+"/experiment"+str(id+1)+".txt", nvars, delimiter=",")
	
def Standardize(variables):
	#variables=DataCleaner().CleanMat(nvariables);
	mu_ = np.mean(variables,axis=0);
	sdev_ = np.std(variables,axis=0);
	nvariables = (variables - mu_) / sdev_;
	n_vars=DataCleaner().CleanMat(nvariables,3); return n_vars;
	#np.savetxt("./experiment99.txt", n_vars, delimiter=",")
	return nvariables;

def Normalize(variables):
	max_ = np.amax(variables,axis=0);
	min_ = np.amin(variables,axis=0);
	denom = 1 / (max_-min_);
	nvariables=(variables - min_)* denom;
	print(np.amax(nvariables,axis=0));
	print(np.amin(nvariables,axis=0));
	n_vars=DataCleaner().CleanMat(nvariables); 
	np.savetxt("./experiment99.txt", n_vars, delimiter=","); return n_vars;
	return nvariables;

def GetRandom(vars,ratio=1,count=0):
	return vars;
	size = vars.shape[0];
	if count==0:
		req = size* ratio;
	else:
		req = count;
	filtered = vars[np.random.choice(size, req, replace=False), :];
	
	return filtered;
#***********************PLOTTING FUNCTIONS*********************************#
	
def EzPlot3d(Nodes,i,j,k):
	plt.ioff();
	
	Plot3d(Nodes[i].GetData(),Nodes[j].GetData(),Nodes[k].GetData());

def Plot3d(source,target,target2):
	plt.ioff();
	fig=plt.figure()
	plt.clf();
	
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(source,target,target2);
	
	plt.show();
	plt.clf();
	plt.close();

def EzPlot2d(Nodes,i,j,counter,path_,save=False,rejected=False):
	#rejected=False;
	plt.ioff();
	if counter==99:
		rejected=True;
	
	
	fname=str(counter)+"_"+str(i)+"_"+str(j)+".png";
	Plot2d(Nodes[i].GetData(),Nodes[j].GetData(),path_+fname,save,rejected);

def EzPlotPlain(Nodes,i,j):
	plt.ioff();
	source = Nodes[i].GetData();
	target = Nodes[j].GetData();
	plt.figure()
	plt.clf();
	plt.plot(source,target,'b.');
	plt.show();
	plt.clf();
	plt.close();
	
def Plot2d(source,target,fname,save=False,rejected=False):
	plt.ioff();
	plt.figure()
	plt.clf();
	
	if not rejected:
		plt.plot(source,target,'b.');
		
	else:
		plt.plot(source,target,'r.');
		
	
	if save:
		plt.savefig(fname);
	#plt.show();
	plt.clf();
	plt.close();
	
def Plot2dL(data1,data2,fname,isPRC=False):
	
	plt.ioff();
	#plt.figure()
	fig, axs = plt.subplots(1, 2)
	#plt.clf();
	
	
	
	if isPRC:
		ylabel='Precision';
		xlabel='Recall';
	else:
		ylabel='Sensitivity';
		xlabel='1-Specificity';
	
	
	

	target =np.array(data1[0]);
	source = np.array(data1[1]);
	auc=data1[2];
	axs[0].set_aspect('equal')
	axs[0].plot(source,target,'go-');
	axs[0].set_xlabel(xlabel);
	axs[0].set_xlim(-0.1,1.1)
	axs[0].set_ylim(-0.1,1.1)
	axs[0].set_ylabel(ylabel);
	axs[0].set_title('AUC Directed: '+str(np.round(auc,6)));
	
	#draw first subplot
	
	target = np.array(data2[0]);
	source = np.array(data2[1]);
	auc=data2[2];
	axs[1].set_aspect('equal')
	axs[1].plot(source,target,'bo-');
	axs[1].set_xlabel(xlabel);
	axs[1].set_ylabel(ylabel);
	axs[1].set_xlim(-0.1,1.1)
	axs[1].set_ylim(-0.1,1.1)
	axs[1].set_title('AUC Undirected: '+str(np.round(auc,6)));
	
	#plt.show();
	fig.savefig(fname);
	plt.clf();
	plt.close();
	
	
def Plot2dL2(source,target,fname,save=False):
	plt.ioff();
	plt.figure()
	plt.clf();
	
	
	plt.xlabel('edge#');
	plt.ylabel('bits saved');
	
	plt.plot(source,target,'b-');
	plt.plot(source,target,'ro');
	

	if save:
		plt.savefig(fname);

	plt.clf();
	plt.close();

def LoadHeader(fname):
	alpha={};
	rev_alpha={};
	i=0;
	print ('Probing: ',fname);
	
	if not os.path.exists(fname):
		for k in range(101):
			alpha[k]=str(k+1);
			rev_alpha[alpha[k]]=k;
		print ('Header file not found, using numbering...');
		return alpha,rev_alpha;
				
	with open(fname,'r') as file:
		k = file.readlines();
		for i in range(len(k)):
			alpha[i]=k[i].strip();
			rev_alpha[alpha[i]]=i;
	#import ipdb;ipdb.set_trace();
	return alpha,rev_alpha;

def LoadGroundTruth(fname,id=11):
	alpha=[];
	i=0;
	print ('Probing: ',fname);
	
	if not os.path.exists(fname):
		print ('Ground Truth file not found, nothing to compare');
		return alpha,False;
				
	with open(fname,'r') as file:
		k = file.readlines();
		for i in range(len(k)):
			if id==11:
				line_ = k[i].split(' ');
			else:
				line_ = k[i].split('\t');
			#print(k[i]);
			s=line_[0].strip();
			t=line_[1].strip();
			alpha.append((s,t));
			
	
	#import ipdb;ipdb.set_trace();
	return alpha,True;




#************************INTERFACE FUNCTIONS*****************************#	
def AggregateGraphs(Graphs, Undecided,count,dims):
	cutoff= count * (1.0/3.0);
	
	Final_graph = [[None for x in range(dims)] for y in range(dims)];
	Ud_graph ={};
	v1=0;
	c1=0;
	
	for k in range(dims):
		for j in range(dims):
			v1=0;
			c1=0;
			size_=1;
			Final_graph[k][j]=(True, size_,Graphs[k][j]);
			
			if (k,j) in Undecided.keys():
				c1+= Graphs[k][j];
				v1+= Undecided[(k,j)]
			
			if v1> c1*0.6:
				Ud_graph[(k,j)]=(True,v1);
			
	
		
	return Final_graph,Ud_graph;

			
def PrintGraph(Final_graph,Nodes,undecided,alpha,dims,logger,base_path="",save=False):
	fname="PredictedGraph.txt";
	
	if save:
		f= open(base_path+fname,"w+")
		f.write('##################\n');
		f.write('Predicted Graph\n');

	
	logger.WriteLog('##################');
	logger.WriteLog('Predicted Graph');
	dims=len(Final_graph);
	counter=0;
	for i in range(dims):
		for j in range(dims):
			if Final_graph[i][j] is not None:
			
				#Plot2d(Nodes[:,j],Nodes[:,i],base_path+str(j)+"_"+str(i)+".png",save);
				if (i,j) in undecided.keys(): # or (j,i) in undecided.keys():
					logger.WriteLog(alpha[j]+ ' --- '+alpha[i]+ ', ('+ str(undecided[(i,j)][1])+')');
				else:
					logger.WriteLog(alpha[j]+ ' --> '+alpha[i]+ ', ('+ str(Final_graph[i][j][1])+')');
					
				if save:
					
					if (i,j) in undecided.keys():# or (j,i) in undecided.keys():
						f.write('{0} {1} 0, ({2})\n'.format(alpha[j],alpha[i],undecided[(i,j)][1]));
					else:
						f.write('{0} {1} 1, ({2})\n'.format(alpha[j],alpha[i],Final_graph[i][j][1]));
					
					
				counter+=1;
	
	if save:
		f.write('Total Edges: {0}\n'.format(counter));
		f.write('##################\n');
		f.close();
	logger.WriteLog('Total Edges: '+str(counter));
	logger.WriteLog('##################');

	
def getAverage(list):
		if len(list)==0:
			return 0;
			
		avg=np.average(np.array(list));
		return avg;
		
def PrintGraph2(Final_graph,Nodes,alpha,dims,logger,base_path="",save=False, param=True):
	if param:
		opt="_poly";
	else:
		opt="_spline";
		
	fname="PredictedGraph"+opt+".txt";
	imname=base_path+"edges_gain.png";
	if save:
		f= open(base_path+fname,"w+")
		f.write('##################\n');
		f.write('Predicted Graph\n');

	g=[];
	for i in range(dims):
		for j in range(dims):
			if i!=j:
				heappush(g,(Final_graph[i][j],(i,j)));
			else:
				heappush(g,(0,(i,j)));
	
	logger.WriteLog('##################');
	logger.WriteLog('Predicted Graph');
	dims=len(Final_graph);
	arr_x=[];
	arr_y=[];
	c=0;
	while len(g)>0:
			v = heappop(g);
			
			i = v[1][0];
			j = v[1][1];
			arr_x.append(c);
			arr_y.append(-v[0]);
			
			c+=1;
			
			#Plot2d(Nodes[:,j],Nodes[:,i],base_path+str(j)+"_"+str(i)+".png",save);
			logger.WriteLog(alpha[j]+ ' --> '+alpha[i]+ ', ('+ str(v[0])+',1)');
			
			if v[0]<0:
				source,target = DataCleaner().Clean(Nodes[:,j],Nodes[:,i],2);
				#Plot2d(source,target,base_path+str(c)+"_"+str(j)+"_"+str(i)+".png",save)
			if save:
				f.write('{0} {1}: ({2},{3})\n'.format(alpha[j],alpha[i],v[0],1));
	
	#Plot2dL2(np.array(arr_x),np.array(arr_y),imname,True);			
	if save:
		#f.write('Total Edges: {0}\n'.format(counter));
		f.write('##################\n');
		f.close();
	#logger.WriteLog('Total Edges: '+str(counter));
	logger.WriteLog('##################');
