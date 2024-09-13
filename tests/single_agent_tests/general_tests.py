import unittest
import pandas as pd
from common.utils import read_adjacency_matrix
from single_agent.influence_maximization import InfluenceMaximization

class GeneralTests(unittest.TestCase):

    def test_BigTestData(self):
        # Test BigTestData
        df = pd.DataFrame()
        g = read_adjacency_matrix('../data/BigTestData.txt')
        algo = ['degdis']
        for a in algo:
            im = InfluenceMaximization(g, 'Agent_0', 40, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        print(df)

    def test_network(self):
        # Test network datatset
        df = pd.DataFrame()
        g = read_adjacency_matrix('../../data/network.txt')
        algo = ['celfpp', 'celf']
        for a in algo:
            im = InfluenceMaximization(g, 'Agent_0', 10, alg=a, diff_model='ic', inf_prob='uniform', r=1000)
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
        print(df)


if __name__ == '__main__':
    unittest.main()
