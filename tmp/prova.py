import networkx as nx
import random

g1 = nx.DiGraph()
g2 = nx.DiGraph()
g1.add_nodes_from([1,2,3,4])
g1.add_edges_from([(1,2),(1,3),(2,4),(3,4)])
nodes = [(node, {"status": "Inactive", "label": None}) for node in g1.nodes]
g2.add_nodes_from(nodes)
edges = []
for (source, target) in g1.edges:
    p = random.random()
    edges.append((source, target, {'weight': p}))
g2.add_edges_from(edges)
print(g1.nodes(data=True))
print(g1.edges(data=True))
print(g2.nodes(data=True))
print(g2.edges(data=True))