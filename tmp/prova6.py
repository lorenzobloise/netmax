from collections import OrderedDict
import networkx as nx
import random
from im_graph import IMGraph
import utils

"""
Tried to combine out-degree heuristic with seed-in-different-communities heuristic, but the performances were worse
"""

g = nx.read_gml('../data/Network for IC-u LT-u.gml',destringizer=int)
for (n,attr) in g.nodes(data=True):
    attr['threshold'] = random.random()
communities = nx.community.louvain_communities(g)
communities_sizes = [len(c) for c in communities]
communities_sizes.sort()
communities_median_size = communities_sizes[round(len(communities_sizes)/2)]
big_communities = []
for c in communities:
    if len(list(c)) >= communities_median_size:
        big_communities.append(c)
rankings = {}
for i in range(len(communities)):
    rankings[i] = list(OrderedDict(sorted(dict(g.out_degree(list(communities[i]))).items(), key=lambda item: item[1], reverse=True)).keys())
num_seed = 100
seed = []
i = 0
while len(seed) != num_seed:
    curr_community = communities[i]
    seed.append(rankings[i].pop(0))
    i = (i+1) % len(big_communities)

imgraph = IMGraph(g,1,utils.DiffusionModel.IC.value,utils.InfluenceProbability.UNIFORM.value, preproc=False, invert_edges=False)
a = "Agent_0"
res = imgraph.diff_model.activate(imgraph, a, seed)
print(res)