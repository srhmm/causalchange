import itertools;
import numpy as np;
from .edge import Edge;
from queue import PriorityQueue;
from copy import deepcopy;
from .graphUtil import GraphUtil;
from .edge_ranking import Scorer;
import gc;

class DAG:
	def __init__ (self,glb,nds,fg,edges,alp,lg,pq,ri):
		self.tolerance=1.0;
		self.prune_tolerance=0.0;
		self.gu = GraphUtil();
		self.alpha = 1e-3;
		self.globe_= glb;
		self.Nodes=nds;
		self.Final_graph=fg;
		self.Edges=edges;
		self.priority_queue=pq;
		self.reverse_index=ri;
		self.save_results=False;
		self.logger=lg;
		self.dict=alp;
		self.dims=len(self.Final_graph);
		self.flip=0;
		self.limms = max(20,0.01 * (((self.dims*self.dims)-self.dims)/2));
		self.parametric=False;
			
	def ForwardSearch(self):
		dict_=self.dict;
		skips=0
		while not self.priority_queue.empty() :
			gc.collect()
			best_edge = self.priority_queue.get();										#load the edge into its respective variables
			edge_cost = best_edge[0];
			node_tuple  = best_edge[1];
			source_node = node_tuple[1]; 
			target_node = node_tuple[0];
			updated_bits = node_tuple[2];
			e=self.Edges[target_node][source_node];										#get the candidate edge between nodes
			#import ipdb; ipdb.set_trace();
			self.logger.WriteLog('Considering Edge from : '+str(source_node)+ ' to: '+ str(target_node)+ ' costing: '+ str(edge_cost));
			#------------------------------SANITY CHECKS-----------------------------------#
			if edge_cost>0:																					#meaning that left over edges are not giving us gain anymore
				break;
			
			flags=[False,False,False];
			flags[0] = ((target_node,source_node) not in self.reverse_index); 								#the node was already added and therefore its entry deleted from reverse index
			flags[1] = not flags[0] and (self.reverse_index[(target_node,source_node)][1] != edge_cost); 	#means the popped queue node is outdated one because self.reverse_index has the cost from latest update
			flags[2] = self.gu.CausesCycle(self.Final_graph,target_node,source_node); 						#will the addition of this edge cause a cycle in graph?
			
			if flags[0]:
				self.logger.WriteLog("Edge already part of the graph");
			
			if flags[1]:
				self.logger.WriteLog("stale node");
				
			if flags[2]:
				self.logger.WriteLog("Edge is cyclic");
				
			if any(flags):	
				if skips>self.limms:																		#Early termination, this tendsto work with very large graphs where the Priority Queue may become full of stale and non-profitable nodes
					break
				skips=skips+1;
				continue;
			
			#---------------------------POPULATE PARENTS-----------------------------------#
			current_parents=[i for i,x in enumerate(self.Final_graph[target_node]) if x is not None];	#populate already existing parents
			current_edges= [];
			parent_nodes=[];
			for current_parent in current_parents:
				current_edges.append(self.Final_graph[target_node][current_parent]);
				parent_nodes.append(self.Nodes[current_parent]);
			
			#---------PVALUE TEST BEGIN (Ref subsection Forward Search in Section Algorithm--------#
			candidate_parent = self.Nodes[source_node];
			child = self.Nodes[target_node];
			candidate_edge = deepcopy(e);
			k = max(np.array([0]),self.reverse_index[(target_node,source_node)][2]);
			pvalue = 2**(-k);
			
			if pvalue > self.alpha:
				self.logger.WriteLog('Pvalue test unsuccessful...this edge is rejected. k='+str(k));
				skips=skips+1;
				continue;
			#---------------------------PVALUE TEST END-----------------------------------#
			
			#---------------------------UPDATE GRAPH--------------------------------------#
			skips=0;																						#found a useful edge, reset early termination counter
			if 	self.Final_graph[target_node][source_node] is None:
				self.Final_graph[target_node][source_node]=deepcopy(e);										#add edge to final_graph
				self.Final_graph[target_node][source_node].score=np.array([edge_cost]);						#update edge score
				self.Edges[target_node][source_node]=Edge(-1,[],np.array([0]),0);							#remove_from set of candidate edges
			else:
				print ('Attempted to rewrite on an old edge: ',e.GetFunctionId());
				continue;																					#This never happens anymore, but I still leave this check here in case there is a corner case that is discovered later on.
				
			self.Nodes[target_node].SetCurrentBits(updated_bits);											#update current bits of target_node
			parent_nodes.append(self.Nodes[source_node]);													#add this node to parents list now
			current_edges.append(self.Final_graph[target_node][source_node]);

			del self.reverse_index[(target_node,source_node)]												#the edge was added, no need to maintain its values anymore
			
			self.logger.WriteLog('*******Added edge from '+str(dict_[source_node])+' to '+str(dict_[target_node])+' with cost: '+str(edge_cost));
			self.logger.WriteLog('Function Id was: '+str(self.Final_graph[target_node][source_node].GetFunctionId()));
			
			#---------------------------PARENT UPDATE-------------------------------------#
			other_parents = [i for i,x in enumerate(self.Edges[target_node]) if x.GetFunctionId() >=0];
			for candidate in other_parents:
				candidate_parent=self.Nodes[candidate];
				gained_ratio,updated_cost,absolute_gain,fid,coeff= self.globe_.GetEdgeAdditionCost(parent_nodes,candidate_parent,child,current_edges);
				
				if (candidate,target_node) not in self.reverse_index:											#no need to update costs for already added parents
					continue;
					
				old_directional_gain = self.reverse_index[(candidate,target_node)][2];
				old_gain_ratio = self.reverse_index[(candidate,target_node)][0];
				v1,v2=self.getIndividualScores(absolute_gain,old_directional_gain,gained_ratio,old_gain_ratio);	#calculate the difference in compression when candidate parent is introduced
					
				if gained_ratio < 1 :																			#if the new parent compresses better, add the updated version to priority queue
					self.priority_queue.put((v1,(target_node,candidate,updated_cost)));
					self.priority_queue.put((v2,(candidate,target_node,candidate_parent.GetCurrentBits()-old_directional_gain)));
					
				self.Edges[target_node][candidate]=Edge(fid,coeff,absolute_gain,v1);							#always update the current Edge-set and the reverse index so stale nodes can be identified
				self.Edges[candidate][target_node].score=v2;
				self.reverse_index[(target_node,candidate)]=(gained_ratio,v1,absolute_gain);
				self.reverse_index[(candidate,target_node)]=(old_gain_ratio,v2,old_directional_gain);

			#---------------------------CHILDREN UPDATE--(Implementation of the Edge-Flip scenario)----------------------#
				current_children=[];
				dims = len(self.Final_graph);
				for i in range(dims):
					if self.Final_graph[i][target_node] is not None:
						current_children.append(i);
				
				for candidate in current_children:
					candidate_parent=self.Nodes[candidate];
					gained_ratio,updated_cost,absolute_gain,fid,coeff = self.globe_.GetEdgeAdditionCost(parent_nodes,candidate_parent,child,current_edges);
					
					ncb,old_directional_gain,old_gain_ratio = self.NetChangeInBits(absolute_gain,self.Final_graph,candidate,target_node);
					
					#if reversing the direction gives us a gain in bits, re-add the current edge as candidate edge to queue (Ref: Section Algorithm in published work)
					if gained_ratio<1.0 and ncb>0:
						v1,v2=self.getIndividualScores(absolute_gain,old_directional_gain,gained_ratio,old_gain_ratio);
					
						activated_edge = deepcopy(self.Final_graph[candidate][target_node]);					
						activated_edge.score=v2;
						self.Final_graph[candidate][target_node]=None										#remove the current edge from graph and add that again as candidate edge
						self.Edges[candidate][target_node]=activated_edge;
						self.Edges[target_node][candidate]=Edge(fid,coeff,absolute_gain,v1);				#update reverse edge statistics for the child edge
						
						self.reverse_index[(target_node,candidate)]=(gained_ratio,v1,absolute_gain);
						self.reverse_index[(candidate,target_node)]=(old_gain_ratio,v2,old_directional_gain);
						
						self.UpdateChildCost(candidate);
						self.logger.WriteLog('#############Flipped edge from: '+str(self.dict[candidate]));
		self.logger.WriteLog("total flips: "+str(self.flip))
		
	#This score is anologous to the score PSI defined in the Edge ranking subsection in the Algorithm section of the publication
	def getIndividualScores(self,absolute_gain,old_directional_gain,gain_ratio,old_gain_ratio):
		S=np.abs(absolute_gain - old_directional_gain);
		score = -(S) 
		if absolute_gain > old_directional_gain:
			v1 = score;
			v2 = -score;
		else:
			v1=-score;
			v2=score;
		return v1,v2;
		
	#This function measures the number of bits we would gain/lose if we flip the direction of edge between the target and the parent node
	#This step is required for the edge-flip scenario that is mentioned in the description of the Forward Search phase	
	def NetChangeInBits(self,absolute_gain,graph,target_node,parent):
		e=graph[target_node][parent];
		graph[target_node][parent]=None;
		
			
		current_parents=[i for i,x in enumerate(graph[target_node]) if x is not None];
		current_edges= [];
		parent_nodes=[];
		for current_parent in current_parents:
			current_edges.append(graph[target_node][current_parent]);
			parent_nodes.append(self.Nodes[current_parent]);
		
		child = self.Nodes[target_node];
		gain_in_bits,new_bits,coeff,absolute_loss=self.globe_.GetCombinationCost(parent_nodes,current_edges,child);
		
		nc=absolute_gain  + absolute_loss;
		
		ratio_loss = (child.GetCurrentBits() /new_bits);
		graph[target_node][parent]=e;
		return nc,-absolute_loss,ratio_loss;
	
	#This function is called once it was concluded that flipping edge between one of the parents of the child node was giving us a gain
	#In this function we calculate the new cost of the child node once the edge from the parent is removed and added back to the priorty q.
	def UpdateChildCost(self,target_node):
		current_parents=[i for i,x in enumerate(self.Final_graph[target_node]) if x is not None];
		current_edges= [];
		parent_nodes=[];
		for current_parent in current_parents:
			current_edges.append(self.Final_graph[target_node][current_parent]);
			parent_nodes.append(self.Nodes[current_parent]);
		
		child = self.Nodes[target_node];
		gain_in_bits,new_bits,coeff,absolute_gain_in_bits=self.globe_.GetCombinationCost(parent_nodes,current_edges,child);
		child.SetCurrentBits(new_bits);

		#---------------------------PARENT UPDATE For the Child Node-------------------------------------#
		other_parents = [i for i,x in enumerate(self.Edges[target_node]) if x.GetFunctionId() >=0];
		for candidate in other_parents:
			candidate_parent=self.Nodes[candidate];
			gained_ratio,updated_cost,absolute_gain,fid,coeff= self.globe_.GetEdgeAdditionCost(parent_nodes,candidate_parent,child,current_edges);
			
			if (candidate,target_node) not in self.reverse_index:
				continue;
			
			old_directional_gain = self.reverse_index[(candidate,target_node)][2];
			old_gain_ratio = self.reverse_index[(candidate,target_node)][0];
			v1,v2=self.getIndividualScores(absolute_gain,old_directional_gain,gained_ratio,old_gain_ratio);
				
			if gained_ratio < 1 :
				self.flip=self.flip+1;
				self.priority_queue.put((v1,(target_node,candidate,updated_cost)));
				self.priority_queue.put((v2,(candidate,target_node,candidate_parent.GetCurrentBits()-old_directional_gain)));
				
			self.reverse_index[(target_node,candidate)]=(gained_ratio,v1,absolute_gain);
			self.reverse_index[(candidate,target_node)]=(old_gain_ratio,v2,old_directional_gain);
			self.Edges[target_node][candidate]=Edge(fid,coeff,absolute_gain,v1);
			self.Edges[candidate][target_node].score=v2;
			
		
	def BackwardSearch(self):
		self.logger.WriteLog('##################');
		self.logger.WriteLog('Begin Pruning....');
		
		for target_node in range(len(self.Nodes)):
			#get all current parents
			edge_removed = True;
			self.logger.WriteLog('Considering Node: '+str(self.dict[target_node]));

			#repeat as long as one of the edges was removed
				#consider all sets of size 1 less
				#if any set gives cost improvement remove that edge
			while edge_removed:
				edge_removed=False;
				current_parents=[i for i,x in enumerate(self.Final_graph[target_node]) if x is not None];
				
				if len(current_parents) <1:
					#print 'Less than 1 parents. Nothing to check, moving on...'
					continue;
				
				set_size = len(current_parents) - 1;
				sets = itertools.combinations(current_parents,set_size);
				
				best_ratio = 1.0;
				best_set = None;
				best_cost = 9999;
				
				for set_ in sets:
					current_edges= [];
					parent_nodes=[];
					for current_parent in set_:
						current_edges.append(self.Final_graph[target_node][current_parent]);
						parent_nodes.append(self.Nodes[current_parent]);
					
					child = self.Nodes[target_node];
					
					gained_ratio,updated_cost,coeff,absolute_gain= self.globe_.GetCombinationCost(parent_nodes,current_edges,child);
					
					if gained_ratio < 1 and gained_ratio<best_ratio:
						best_ratio=gained_ratio;
						best_set = set_;
						best_cost = updated_cost
						best_absolute_gain = absolute_gain;
						self.logger.WriteLog('Best edge set updated to: '+str(set_));
						edge_removed=True;
						
					
				if edge_removed:
					#remove the value from current set
					master_set = set(current_parents);
					sub_set = set(best_set);
					removed_index = master_set - sub_set;
					removed_index = next(iter(removed_index));
					self.logger.WriteLog('Removing: '+str(self.dict[removed_index]));
					self.Final_graph[target_node][removed_index]=None;
					self.logger.WriteLog(str(self.dict[target_node])+ ' cost update to ' +  str(best_cost)+ ' from '+str(self.Nodes[target_node].GetCurrentBits()));
					self.Nodes[target_node].SetCurrentBits(best_cost);
			
			self.logger.WriteLog('Finished Node: '+str(self.dict[target_node]));
			self.logger.WriteLog('***************************');

