from collections import OrderedDict
import networkx as nx

import diffusion_models
import influence_probabilities
import utils
from im_graph import IMGraph
from tqdm import tqdm
import random

"""
Confronting heuristic of choosing the node with maximum in-degree/out-degree/betweenness respectively.
Found out that the best heuristic is choosing the node with maximum out-degree
"""

n_nodes = [25,50]
conn_prob = [0.1,0.2]

print(f"Number of nodes range: {n_nodes}")
print(f"Probability of connections range: {conn_prob}")

in_total = 0
out_total = 0
betweenness_total = 0

num_trials = 10000

print(f"Number of trials: {num_trials}")

for _ in tqdm(range(num_trials), desc="Completed trials"):
    g = nx.erdos_renyi_graph(random.randint(n_nodes[0],n_nodes[1]), random.uniform(conn_prob[0],conn_prob[1]), directed=True)
    in_degree_ranking = OrderedDict(sorted(dict(g.in_degree()).items(), key=lambda item: item[1], reverse=True))
    in_max = next(iter(in_degree_ranking))
    out_degree_ranking = OrderedDict(sorted(dict(g.out_degree()).items(), key=lambda item: item[1], reverse=True))
    out_max = next(iter(out_degree_ranking))
    betweenness_ranking = OrderedDict(
        sorted(dict(nx.betweenness_centrality(g)).items(), key=lambda item: item[1], reverse=True))
    betweenness_max = next(iter(betweenness_ranking))
    imgraph = IMGraph(g,1,diffusion_models.Triggering(influence_probabilities.Uniform()),insert_prob=True)
    a = "Agent_0"
    res_in = imgraph.diff_model.activate(imgraph, a, [in_max])
    in_total += res_in
    res_out = imgraph.diff_model.activate(imgraph, a, [out_max])
    out_total += res_out
    res_betweenness = imgraph.diff_model.activate(imgraph, a, [betweenness_max])
    betweenness_total += res_betweenness

print(f"Average seed set size with in-degree heuristic: {in_total/num_trials}")
print(f"Average seed set size with out-degree heuristic: {out_total/num_trials}")
print(f"Average seed set size with betweenness heuristic: {betweenness_total/num_trials}")