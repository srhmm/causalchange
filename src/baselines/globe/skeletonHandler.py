from .edge import Edge;
from queue import PriorityQueue
import numpy as np;
from .edge_ranking import Scorer;
import gc;


class SkeletonHandler:

	def __init__(self,slp,glb,lg):
		self.slope_=slp;
		self.globe_=glb;
		self.logger=lg;
		self.q = PriorityQueue();
		
	def RankEdges(self,undirected_edges,Nodes,Edges,Final_Graph):
		#The reverse index is to reverse reference the entries of the priority queue. This will help us later in checking if the priority queue entries are stale
		reverse_index = {};		
		cc=0;

		#go through each entry
		for edge in undirected_edges:
			gc.collect()
			i = edge[0];
			j = edge[1];
			cc+=1;

			if True and cc%50==0:
				print(str(cc)+" edges out of "+ str(len(undirected_edges))+" ranked...")

			#Rank edge in both possible directions
			gain_ratio1,best_new_bits1,x1_best_absolute1,best_fids1,coeffs1=self.globe_.GetEdgeAdditionCost([],Nodes[j],Nodes[i],[]);
			gain_ratio2,best_new_bits2,x2_best_absolute2,best_fids2,coeffs2=self.globe_.GetEdgeAdditionCost([],Nodes[i],Nodes[j],[]);
				
			
			#x1_best_absolute is analogous to the delta function described in the Algorithm section
			#x2_best_absolute is analogous to the delta function described in the Algorithm section
			#S is the PSI function described in the Algorithm section
			S = np.abs(x1_best_absolute1-x2_best_absolute2)
			score_ = -(S)
			
			self.logger.WriteLog("Ranked Edge between Nodes: "+str(i)+" and "+str(j));
			
			if x1_best_absolute1 > x2_best_absolute2:
				v1=score_;
				v2=-score_;
			else:
				v1=-score_;
				v2=score_;
			
			self.logger.WriteLog(str(gain_ratio1)+": "+str(x1_best_absolute1)+","+str(x2_best_absolute2));
			e1 = Edge(best_fids1,coeffs1,x1_best_absolute1,v1);
			e2 = Edge(best_fids2,coeffs2,x2_best_absolute2,v2);

			Edges[i][j] = e1;
			reverse_index[(i,j)] = (gain_ratio1,v1,x1_best_absolute1)
			if gain_ratio1 < 1:
				self.logger.WriteLog("Added to q: "+str(v1)+" : "+str(best_new_bits1));
				self.q.put( (v1,(i,j,best_new_bits1)));
			
			Edges[j][i] = e2;
			reverse_index[(j,i)] = (gain_ratio2,v2,x2_best_absolute2);
			if gain_ratio2<1:
				self.logger.WriteLog("Added to q: "+str(v2)+" : "+str(best_new_bits2));
				self.q.put((v2,(j,i,best_new_bits2)));

		
		return self.q,reverse_index;

		
	
			
		
		
