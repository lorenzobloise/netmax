from collections import OrderedDict
import random
import networkx as nx
from im_graph import IMGraph
import utils

"""
Tried to implement the seed-no-neighbors heuristic, but the performances were worse
"""

g = nx.DiGraph()
g.add_nodes_from([1,2,3,4,5,6,7])
g.add_edges_from([(1,2),(1,5),(1,6),(2,3),(2,6),(3,6),(4,3),(6,1),(6,4),(6,5),(6,7),(7,4)])

g2 = nx.DiGraph()
g2.add_nodes_from([
    ('a', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.2}),
    ('b', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.3}),
    ('c', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.4}),
    ('d', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.8}),
    ('e', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.6}),
    ('f', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.4}),
    ('g', {'status': 'INACTIVE', 'agent': 'NONE', 'threshold': 0.3})
])
g2.add_edges_from([
    ('a','b',{'p': 0.4}),
    ('a','e',{'p': 0.2}),
    ('a','f',{'p': 0.3}),
    ('b','c',{'p': 0.6}),
    ('b','f',{'p': 0.1}),
    ('c','f',{'p': 0.1}),
    ('d','c',{'p': 0.25}),
    ('f','a',{'p': 0.05}),
    ('f','d',{'p': 0.2}),
    ('f','e',{'p': 0.1}),
    ('f','g',{'p': 0.3}),
    ('g','d',{'p': 0.02})
])

g3 = nx.read_gml('../data/Network for IC-u LT-u.gml',destringizer=int)
for (n,attr) in g3.nodes(data=True):
    attr['threshold'] = random.random()
out_degree_ranking = OrderedDict(sorted(dict(g3.out_degree()).items(), key=lambda item: item[1], reverse=True))

imgraph = IMGraph(g3,1,utils.DiffusionModel.IC.value,utils.InfluenceProbability.UNIFORM.value, preproc=False, invert_edges=False)

a = "Agent_0"
num_seed = 100
seed = []
ranking = list(out_degree_ranking.keys())
copy = list(out_degree_ranking.keys())

"""
seed = []
copy = ranking.copy()
for i in range(num_seed):
  if ranking.size() < num_seed:
    seed.append(copy.pop())
  else:
    candidate = ranking.pop() # Prendi il primo
    seed.append(candidate)
    copy.delete(candidate)
    neighbors = candidate.neighbors() # In-neighbor e out-neighbor
    for n in neighbors:
      ranking.delete(n)
"""
for i in range(num_seed):
    if len(ranking) < num_seed:
        seed.append(copy.pop(0))
    else:
        candidate = ranking.pop(0)
        seed.append(candidate)
        copy.remove(candidate)
        in_neighbors = [u for (u,v) in imgraph.graph.in_edges(candidate)]
        out_neighbors = [v for (u,v) in imgraph.graph.out_edges(candidate)]
        neighbors = list(set(in_neighbors + out_neighbors))
        for n in out_neighbors:
            if n in ranking:
                ranking.remove(n)

res = imgraph.diff_model.activate(imgraph, a, seed)
print(res)