from heapq import heappush, heappop
from copy import deepcopy
from sklearn.metrics import auc
from queue import PriorityQueue
import numpy as np;
class StatsCalculator:

	def MainStats(self, Graph,truth,headers):
		dims= len(Graph);
		
		total=0;
		correct=0;
		#iterate over whole graph
		for i in range(dims):
			for j in range(dims):
				
				val = Graph[i][j]; #0 indicates edge was not predicted
				if i!=j and val<-0.1:
					tuple_ = (i,j);#extract the edge
					total=total+1;#add to total
					source_= headers[tuple_[1]];
					target_= headers[tuple_[0]];
					if (source_,target_) in truth:
						correct= correct+1; #add one to correct if edge is also in truth
		return [total,correct]
				

	def DirectedPRCROC(self, Graph,Truth,Headers):
		#graph is the predictions
		#truth is the groun truth
		
		ordered_graph=[];
		dims= len(Graph);
		unique_thresh=[];
		#first we need to order the graph
		precision=[0];
		recall=[0];
		fpr=[0];
		
		for i in range(dims):
			for j in range(dims):
				tuple_ = (i,j);
				val = Graph[i][j];
				if i!=j:
					heappush(ordered_graph,(val,tuple_));
					if val not in unique_thresh:
							heappush(unique_thresh,val);
					
			
		#now ordered graph should contain predictions in decreasing order
		#for i in range(len(ordered_graph)):

		while len(unique_thresh)>0:
			i = heappop(unique_thresh);
			#print 'Limit count: ', i;
			pr , rec,fps  = self.ComputeDirectedStats(ordered_graph,Truth,Headers,i);
			precision.append(pr);
			recall.append(rec);
			fpr.append(fps);
		
		area_prc=auc(recall,precision);#import ipdb; ipdb.set_trace();
		area_roc=auc(fpr,recall);#import ipdb; ipdb.set_trace();
		
		return [precision,recall,area_prc],[recall,fpr,area_roc];
	
	def ComputeDirectedStats(self,graph,truth,headers,limit):
		graph_ = deepcopy(graph);
		tp=0;
		fp=0;
		tn=0;
		fn=0;
		
		#for i in range(limit):
		v = heappop(graph_);
		source_= headers[v[1][1]];
		target_= headers[v[1][0]];
			
		while v[0] <= limit and len(graph_)>0:
			#print abs(v[0]),' vs ', abs(limit);
			if (source_,target_) in truth:
				tp+=1;
			else:
				fp+=1;
			
			v = heappop(graph_);
			source_= headers[v[1][1]];
			target_= headers[v[1][0]];
		
		flag=False;
		if len(graph_)==0:
			#print abs(v[0]),' vs ', abs(limit);
			flag=True;
			if (v[0] <= limit) and ((source_,target_) in truth):
				tp+=1;
			elif (v[0] > limit) and ((source_,target_) in truth):
				fn+=1;
			elif (v[0] <=limit) and not ((source_,target_) in truth):
				fp+=1;
			else:
				tn+=1;
		
				
		while len(graph_)>0:
			#print abs(v[0]), ' is done for..'
			if (source_,target_) in truth:
				fn+=1;
			else:
				tn+=1;
				
			v = heappop(graph_);
			#import ipdb; ipdb.set_trace();
			source_= headers[v[1][1]];
			target_= headers[v[1][0]];
				
		#print abs(v[0]), ' is done for..'
		if not flag:
			if(source_,target_) in truth:
				fn+=1;
			else:
				tn+=1;

		if tp==0: 
			pr =0; rec=0; tpr=0;
		else:
			pr = (tp*1.0) / 	(tp+fp);
			rec = (tp*1.0) / 	(tp+fn);
		
		if fp==0:
			fpr=0;
		else:
			fpr = (fp*1.0)/(fp+tn);
		return pr,rec,fpr
			

			
	
	def UndirectedPRCROC(self, Graph,Truth,Headers):
		#graph is the predictions
		#truth is the groun truth
		
		ordered_graph=[];
		dims= len(Graph);
		#first we need to order the graph
		precision=[0];
		recall=[0];
		fpr=[0];
		unique_thresh=[];
		
		for i in range(dims):
			for j in range(i+1,dims):
				tuple_ = (i,j);
				val = min(Graph[i][j],Graph[j][i]);
				heappush(ordered_graph,(val,tuple_));
				if val not in unique_thresh:
						heappush(unique_thresh,val);
				
		
		#now ordered graph should contain predictions in decreasing order
		while len(unique_thresh)>0:
			i = heappop(unique_thresh);
			#print 'Limit count: ', i;
			pr , rec,fps  = self.ComputeUndirectedStats(ordered_graph,Truth,Headers,i);
			precision.append(pr);
			recall.append(rec);
			fpr.append(fps);
	
		#import ipdb; ipdb.set_trace();
		area_prc=auc(recall,precision);#import ipdb; ipdb.set_trace();
		area_roc=auc(fpr,recall);#import ipdb; ipdb.set_trace();
		
		return [precision,recall,area_prc],[recall,fpr,area_roc];
			
	def ComputeUndirectedStats(self,graph,truth,headers,limit):
		graph_ = deepcopy(graph);
		tp=0;
		fp=0;
		tn=0;
		fn=0;
		
		v = heappop(graph_);
		first_= headers[v[1][0]];
		second_= headers[v[1][1]];
		#import ipdb; ipdb.set_trace();
		while v[0] <= limit and len(graph_)>0:		
			if (first_,second_) in truth or (second_,first_) in truth:
				tp+=1;
			else:
				fp+=1;
				
			v = heappop(graph_);
		
			first_= headers[v[1][0]];
			second_= headers[v[1][1]];
			#import ipdb; ipdb.set_trace();
		flag=False;
		if len(graph_)==0 :
			flag=True;
			if (v[0]<=limit) and ((first_,second_) in truth or (second_,first_) in truth):
				tp+=1;
			elif (v[0]>limit) and ((first_,second_) in truth or (second_,first_) in truth):
				fn+=1;
			elif (v[0]<=limit) and not ((first_,second_) in truth or (second_,first_) in truth):
				fp+=1;
			else:
				tn+=1;
				
		#import ipdb; ipdb.set_trace();	
		while len(graph_)>0:
			if (first_,second_) in truth or (second_,first_) in truth:
				fn+=1;
			else:
				tn+=1;
			v = heappop(graph_);
			#import ipdb; ipdb.set_trace();
			first_= headers[v[1][0]];
			second_= headers[v[1][1]];
		
		#import ipdb; ipdb.set_trace();		
		if not flag:
			if ((first_,second_) in truth or (second_,first_) in truth):
				fn+=1;
			else:
				tn+=1;

		if tp==0: 
			pr =0; rec=0;
		else:
			pr = (tp*1.0) / 	(tp+fp);
			rec = (tp*1.0) / 	(tp+fn);
		
		
		if fp==0:
			fpr=0;
		else:
			fpr = (fp*1.0)/(fp+tn);
			
		#import ipdb; ipdb.set_trace();
		return pr,rec,fpr
			
			
			