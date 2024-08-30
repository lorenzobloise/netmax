import time
import gc
from tqdm import tqdm
import diffusion_models
import influence_probabilities
import utils
from algorithms import proxy_based, simulation_based
from data.transform_data import read_dataset
from im_graph import IMGraph
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

AGENT = "Agent_0"
BUDGET = 10

diffusion_models = [("IC", diffusion_models.IndependentCascade(influence_probabilities.Uniform()))]

def run_diffusion_model(input_graph, diffusion_model, algorithm, budget, repetitions):
    start_time = time.time()
    imgraph = IMGraph(input_graph, 1, diffusion_model[1])
    # Take seed set
    if algorithm == simulation_based.CELF:
        seed_set = algorithm(AGENT, budget, imgraph, 1000).run()
    else:
        seed_set = algorithm(AGENT, budget, imgraph).run()
    # Run simulation
    spread = utils.simulation(imgraph, AGENT, seed_set, repetitions)
    end_time = time.time() - start_time
    # return a row as result
    result_row = {
        "algorithm": [algorithm.__name__],
        "budget": [budget],
        "time": [end_time],
        "seed": [seed_set],
        "diffusion_model": [diffusion_model[0]],
        "influence_probability": ["Uniform"],
        "spread": [spread]
    }
    return result_row

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def my_similarity(list1, list2):
    number_of_elements_of_list1_in_list2 = 0
    for x in list1:
        if x in list2:
            number_of_elements_of_list1_in_list2 += 1
    return number_of_elements_of_list1_in_list2 / len(list1)

def calculate_similarities(df):
    # Calculate similarities between seed sets of different algorithms
    algorithms = df["algorithm"].unique()
    print(algorithms)
    for i, row in df.iterrows():
        current_seed_set = row["seed"]
        print(current_seed_set)
        for elem in algorithms:
            other_seed_set=df[df["algorithm"] == elem].iloc[0]["seed"]
            df.loc[i, f'similarity_{elem}'] = my_similarity(current_seed_set, other_seed_set)
    print(df)

def __main__():
    df = pd.DataFrame()
    # Run the simulation
    g = read_dataset('../data/network.txt')
    print(f"Number of nodes: {len(g.nodes)}")
    print(f"Number of edges: {len(g.edges)}")
    for diffusion_model in tqdm(diffusion_models, desc="Diffusion Models"):
        algorithms = [proxy_based.Group_PR, proxy_based.DegDis, proxy_based.HighestOutDegree, simulation_based.CELF]
        for algorithm in tqdm(algorithms, desc="Algorithms"):
            row = run_diffusion_model(g, diffusion_model, algorithm, BUDGET, 1000)
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
            gc.collect()
    print(df)
    calculate_similarities(df)


__main__()