import networkx as nx
from im_graph import IMGraph
import utils
from algorithms import simulation_based
import matplotlib.pyplot as plt

g = nx.erdos_renyi_graph(100, 0.03, directed=True)
print(f"Number of nodes: {len(g.nodes)}")
print(f"Number of edges: {len(g.edges)}")
imgraph = IMGraph(g,1,utils.DiffusionModel.IC.value,preproc=True,invert_edges=False)
agent = "Agent_0"
budget = 10
r = 10000

(mcgreedy_seed_size, mcgreedy_spreads, mcgreedy_num_simulations, mcgreedy_computation_time) = algorithms.MCGreedy(agent, budget, imgraph, r).run()
(celf_seed_size, celf_spreads, celf_num_simulations, celf_computation_time) = algorithms.CELF(agent, budget, imgraph, r).run()

print("")
print(f"MCGREEDY\n\nSeed set size: {mcgreedy_seed_size}\nExpected spreads: {mcgreedy_spreads}\nNumber of simulations: {mcgreedy_num_simulations}\nTimelapse: {mcgreedy_computation_time}")
print("")
print(f"CELF\n\nSeed set size: {celf_seed_size}\nExpected spreads: {celf_spreads}\nNumber of simulations: {celf_num_simulations}\nTimelapse: {celf_computation_time}")

plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.plot(range(0,len(mcgreedy_computation_time)),mcgreedy_computation_time,label="MCGreedy",color="#FBB4AE")
plt.plot(range(0,len(celf_computation_time)),celf_computation_time,label="CELF",color="#B3CDE3")
plt.ylabel('Computation Time (Seconds)')

plt.xlabel('Size of Seed Set')
plt.title('Computation Time')
plt.legend(loc=2)

plt.show()

plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.plot(range(0,len(mcgreedy_spreads)),mcgreedy_spreads,label="MCGreedy",color="#FBB4AE")
plt.plot(range(0,len(celf_spreads)),celf_spreads,label="CELF",color="#B3CDE3")
plt.ylabel('Expected Spread')
plt.xlabel('Size of Seed Set')
plt.title('Expected Spread')
plt.legend(loc=2)

plt.show()