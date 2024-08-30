import diffusion_models
from algorithms import proxy_based
import influence_probabilities
from data.transform_data import read_dataset
from im_graph import IMGraph
import time
import pandas as pd
import utils
from algorithms import simulation_based

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

NUM_SIMULATIONS = 10000
AGENT = "Agent_0"
BUDGET = 10

g = read_dataset('../data/network.txt')
print(f"Number of nodes: {len(g.nodes)}")
print(f"Number of edges: {len(g.edges)}")

for (node,attr) in g.nodes(data=True):
    attr['status'] = utils.NodeStatus.INACTIVE.value
    attr['agent'] = utils.NO_AGENT_LABEL

imgraph = IMGraph(g,1,diffusion_models.IndependentCascade(influence_probabilities.Uniform()),preproc=False)

start_time = time.time()

# Out degree heuristic
outdeg_seed = proxy_based.HighestOutDegree(AGENT, BUDGET, imgraph).run()
outdeg_spread_50 = utils.simulation(imgraph, AGENT, outdeg_seed, r=50)
outdeg_spread_1000 = utils.simulation(imgraph, AGENT, outdeg_seed, r=1000)
outdeg_spread_10000 = utils.simulation(imgraph, AGENT, outdeg_seed, r=10000)
outdeg_time = time.time() - start_time

# Degree Discount heuristic
start_time = time.time()
dd_seed = proxy_based.DegDis(AGENT, BUDGET, imgraph).run()
dd_spread_50 = utils.simulation(imgraph, AGENT, dd_seed, r=50)
dd_spread_1000 = utils.simulation(imgraph, AGENT, dd_seed, r=1000)
dd_spread_10000 = utils.simulation(imgraph, AGENT, dd_seed, r=10000)
dd_time = time.time() - start_time

# MCGreedy
start_time = time.time()
mcgreedy_seed = simulation_based.MCGreedy(AGENT, BUDGET, imgraph, NUM_SIMULATIONS).run()
mcgreedy_spread_50 = utils.simulation(imgraph, AGENT, mcgreedy_seed, r=50)
mcgreedy_spread_1000 = utils.simulation(imgraph, AGENT, mcgreedy_seed, r=1000)
mcgreedy_spread_10000 = utils.simulation(imgraph, AGENT, mcgreedy_seed, r=10000)
mcgreedy_time = time.time() - start_time

# CELF
start_time = time.time()
celf_seed = simulation_based.CELF(AGENT, BUDGET, imgraph, NUM_SIMULATIONS).run()
celf_spread_50 = utils.simulation(imgraph, AGENT, celf_seed, r=50)
celf_spread_1000 = utils.simulation(imgraph, AGENT, celf_seed, r=1000)
celf_spread_10000 = utils.simulation(imgraph, AGENT, celf_seed, r=10000)
celf_time = time.time() - start_time

#CELF++
start_time = time.time()
celfpp_seed = simulation_based.CELF_PP(AGENT, BUDGET, imgraph, NUM_SIMULATIONS).run()
celfpp_spread_50 = utils.simulation(imgraph, AGENT, celfpp_seed, r=50)
celfpp_spread_1000 = utils.simulation(imgraph, AGENT, celfpp_seed, r=1000)
celfpp_spread_10000 = utils.simulation(imgraph, AGENT, celfpp_seed, r=10000)
celfpp_time = time.time() - start_time

# Results in pandas dataframe
data = {
    'Algorithm': ['OutDeg', 'DegDis', 'MCGreedy', 'CELF', 'CELF++'],
    'Seed set': [outdeg_seed, dd_seed, mcgreedy_seed, celf_seed, celfpp_seed],
    'Spread 50 sim': [outdeg_spread_50, dd_spread_50, mcgreedy_spread_50,
                      celf_spread_50, celfpp_spread_50],
    'Spread 1000 sim': [outdeg_spread_1000, dd_spread_1000,
                        mcgreedy_spread_1000, celf_spread_1000, celfpp_spread_1000],
    'Spread 10000 sim': [outdeg_spread_10000, dd_spread_10000,
                        mcgreedy_spread_10000, celf_spread_10000, celfpp_spread_10000],
    'Time (s)': [outdeg_time, dd_time, mcgreedy_time, celf_time, celfpp_time]
}

# DataFrame
df = pd.DataFrame(data)

print(df)