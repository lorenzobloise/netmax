from im import IM
from data.transform_data import read_dataset
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

g = read_dataset('../data/BigTestData.txt')
max_budget = 200
spreads = []
for b in tqdm(range(1,max_budget), desc="Test submodularity"):
    im = IM(g, {'Agent_0': b}, alg='degdis', diff_model='ic', inf_prob='uniform', r=10)
    seed = im.run()
    spreads.append(im.result['spread'])
    gc.collect()
plt.plot(range(1,max_budget), spreads)
plt.xlabel('Budget')
plt.ylabel('Spread')
plt.show()