import networkx as nx
import utils
from im_graph import IMGraph
from algorithms.simulation_based import MCGreedy
import random

g2 = nx.DiGraph()
g2.add_nodes_from([1,2,3,4,5,6,7])
g2.add_edges_from([(1,2),(1,5),(1,6),(2,3),(2,6),(3,6),(4,3),(6,1),(6,4),(6,5),(6,7),(7,4)])

g = nx.to_directed(nx.barabasi_albert_graph(100,2))
nodes = list(g.nodes)
edges = []
for (source,target) in g.edges:
    p = random.random()
    if p < 0.5:
        edges.append((source,target))
    else:
        edges.append((target,source))
g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)

#g3 = nx.read_gml('../data/Network for IC-u LT-u.gml',destringizer=int)

print(f"Number of nodes: {len(g2.nodes)}")
print(f"Number of edges: {len(g2.edges)}")
#print(f"In-degree ranking: {OrderedDict(sorted(nx.in_degree_centrality(g).items(), key=lambda item: item[1], reverse=True))}")
imgraph = IMGraph(g2,1,utils.DiffusionModel.IC.value,preproc=True,invert_edges=False)
alg = MCGreedy("Agent_0",5,imgraph,r=1000)
opt_seed_set = alg.run()
print(f"Optimal seed set: {opt_seed_set}")