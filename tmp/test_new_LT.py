import time
from algorithms import proxy_based
import diffusion_models
import influence_probabilities
import utils
from data.transform_data import read_dataset
from im_graph import IMGraph
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame()

g = read_dataset('../data/network.txt')
start_time = time.time()
imgraph = IMGraph(g,1,diffusion_models.IndependentCascade(influence_probabilities.Uniform()))
print(f"Number of nodes: {len(imgraph.graph.nodes)}")
print(f"Number of edges: {len(imgraph.graph.edges)}")
seed_set = proxy_based.Group_PR("Agent_0", 10, imgraph).run()
spread = utils.simulation(imgraph, "Agent_0", seed_set, 1)
end_time = time.time() - start_time
result_row = {
    "time": [end_time],
    "seed": [seed_set],
    "spread": [spread]
}
df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)

print(df)