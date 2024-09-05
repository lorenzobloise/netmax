import unittest
from single_agent.influence_maximization import InfluenceMaximization
import pandas as pd
from common.utils import read_graph, __calculate_similarities__

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class MyTestCase(unittest.TestCase):

    def test_BigTestData(self):
        # Test BigTestData
        df = pd.DataFrame()
        g = read_graph('../data/BigTestData.txt')
        algo = ['cgina']
        for a in algo:
            im = InfluenceMaximization(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        print(__calculate_similarities__(df))

    def test_epinions_d_5(self):
        # Test epinions_d_5
        df = pd.DataFrame()
        g = read_graph('../data/epinions_d_5.txt')
        algo = ['group-pr']
        for a in algo:
            im = InfluenceMaximization(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        print(__calculate_similarities__(df))

    def test_twitter_d(self):
        # Test epinions_d_5
        df = pd.DataFrame()
        g = read_graph('../data/twitter-d.txt')
        algo = ['group-pr']
        for a in algo:
            im = InfluenceMaximization(g, {'Agent_0': 8130}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        print(__calculate_similarities__(df))

    def test_NetHEPT(self):
        # Test NetHEPT
        df = pd.DataFrame()
        g = read_graph('../data/NetHEPT.txt')
        algo = ['group-pr']
        for a in algo:
            im = InfluenceMaximization(g, {'Agent_0': 40}, alg=a, diff_model='ic', inf_prob='uniform', r=1)
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
        print(__calculate_similarities__(df))

    def test_network(self):
        # Test network
        df = pd.DataFrame()
        g = read_graph('../data/network.txt')
        algo = ['simpath','group-pr', 'celf']
        for a in algo:
            im = InfluenceMaximization(g, {'Agent_0': 10}, alg=a, diff_model='lt', inf_prob='uniform', r=1000)
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
        print(__calculate_similarities__(df))


if __name__ == '__main__':
    unittest.main()