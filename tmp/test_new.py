from im import IM, simulation
from data.transform_data import read_dataset
import time
import pandas as pd
import gc

def my_similarity(list1, list2):
    number_of_elements_of_list1_in_list2 = 0
    for x in list1:
        if x in list2:
            number_of_elements_of_list1_in_list2 += 1
    return number_of_elements_of_list1_in_list2 / len(list1)

def calculate_similarities(dataframe):
    # Calculate similarities between seed sets of different algorithms
    algorithms = dataframe["algorithm"].unique()
    for i, row in dataframe.iterrows():
        current_seed_set = row["seed"]
        for elem in algorithms:
            other_seed_set=dataframe[dataframe["algorithm"] == elem].iloc[0]["seed"]
            dataframe.loc[i, f'similarity_{elem}'] = my_similarity(current_seed_set, other_seed_set)
    print(dataframe)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame()
g = read_dataset('../data/network.txt')
algos = ['mcgreedy','celf','celfpp','ublf','cga','outdeg','degdis','ubound','group-pr']
for a in algos:
    im = IM(g, {'Agent_0': 10}, alg=a, diff_model='ic', inf_prob='uniform', r=1000)
    seed = im.run()
    spread = im.result['spread']
    execution_time = im.result['execution_time']
    result_row = {
        "algorithm": [a],
        "time": [execution_time],
        "seed": [seed],
        "spread": [spread]
    }
    df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
    gc.collect()
calculate_similarities(df)