import unittest
from data.transform_data import read_dataset
from im import IM
import pandas as pd
import math


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class MyTestCase(unittest.TestCase):

    def __my_similarity__(self, list1, list2):
        number_of_elements_of_list1_in_list2 = 0
        for x in list1:
            if x in list2:
                number_of_elements_of_list1_in_list2 += 1
        return number_of_elements_of_list1_in_list2 / len(list1)

    def __calculate_similarities__(self,dataframe):
        # Calculate similarities between seed sets of different algorithms
        algorithms = dataframe["algorithm"].unique()
        for i, row in dataframe.iterrows():
            current_seed_set = row["seed"]
            for elem in algorithms:
                other_seed_set=dataframe[dataframe["algorithm"] == elem].iloc[0]["seed"]
                dataframe.loc[i, f'similarity_{elem}'] = self.__my_similarity__(current_seed_set, other_seed_set)
        print(dataframe)

    def test_BigTestData(self):
        # Test BigTestData
        df = pd.DataFrame()
        g = read_dataset('../data/BigTestData.txt')
        algo = ['cgina']
        for a in algo:
            im = IM(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        self.__calculate_similarities__(df)

    def test_epinions_d_5(self):
        # Test epinions_d_5
        df = pd.DataFrame()
        g = read_dataset('../data/epinions_d_5.txt')
        algo = ['group-pr']
        for a in algo:
            im = IM(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        self.__calculate_similarities__(df)

    def test_twitter_d(self):
        # Test epinions_d_5
        df = pd.DataFrame()
        g = read_dataset('../data/twitter-d.txt')
        algo = ['group-pr']
        for a in algo:
            im = IM(g, {'Agent_0': 8130}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        self.__calculate_similarities__(df)

    def test_NetHEPT(self):
        # Test NetHEPT
        df = pd.DataFrame()
        g = read_dataset('../data/NetHEPT.txt')
        algo = ['group-pr']
        for a in algo:
            im = IM(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        self.__calculate_similarities__(df)

    def test_network(self):
        # Test network
        df = pd.DataFrame()
        g = read_dataset('../data/network.txt')
        algo = ['simpath','group-pr', 'celf']
        for a in algo:
            im = IM(g, {'Agent_0': 10}, alg=a, diff_model='lt', inf_prob='uniform', r=1000)
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
        self.__calculate_similarities__(df)

    def test_network_competitive(self):
        df = pd.DataFrame()
        g = read_dataset('../data/network.txt')
        a = 'celf'
        agents = {
            'Agent_0': 5,
            'Agent_1': 5
        }
        im = IM(g, agents, a, 'ic', 'uniform', r=10)
        seed = im.run()
        spread = im.result['spread']
        execution_time = im.result['execution_time']
        result_row = {
            "time": [execution_time],
            "seed": [seed],
            "spread": [spread]
        }
        print(result_row)

if __name__ == '__main__':
    unittest.main()