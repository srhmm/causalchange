from copy import deepcopy
from .edge import Edge;

class GraphUtil:

	def CausesCycle(self,Final_graph,target_node,source_node):
		temp_graph = deepcopy(Final_graph);
		temp_graph[target_node][source_node]=Edge(-1,[],[],0);
		return self.HasCycle(temp_graph);
		
	def HasCycle(self,graph):
		V = len(graph);
		visited = [False] * V;
		onStack = [False] * V;
		
		for n in range(V):
			if not visited[n]:
				if self.CycleChecker(graph,n,visited,onStack) == True:
					return True;
		return False;
	
	def CycleChecker(self,graph,node,visited,stack):
		visited[node]=True;
		stack[node]=True;
		
		neighbours=[i for i,x in enumerate(graph[node]) if x is not None];
		
		for neighbour in neighbours:
			if not visited[neighbour]:
				if self.CycleChecker(graph,neighbour,visited,stack):
					return True;
			elif stack[neighbour]==True:
				return True;
				
		stack[node]=False;
		return False;
		