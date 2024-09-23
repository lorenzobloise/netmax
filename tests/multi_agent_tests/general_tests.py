import unittest
import random
import pandas as pd
from common.utils import read_adjacency_matrix
import influence_maximization as im
from agent import Agent

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class GeneralTests(unittest.TestCase):

    def __create_agents__(self, num_agents):
        agents = {}
        for i in range(num_agents):
            agent_name = 'Agent_' + str(i)
            budget = random.randint(10,10)
            agents[agent_name] = budget
        return agents

    def reset_agents(self, agents):
        for agent in agents:
            agent.seed = []

    def test_network(self):
        # Test Network
        df = pd.DataFrame()
        g = read_adjacency_matrix('../../data/network.txt')
        algo = ['tim', 'tim_p']
        dict_of_agents = self.__create_agents__(num_agents=1)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                     diff_model='ic', inf_prob=None, r=1,
                                                     insert_opinion=False, endorsement_policy='random', verbose=True)
            seed = im_instance.run()
            spread = im_instance.result['spread']
            execution_time = im_instance.result['execution_time']
            result_row = {
                "algorithm": [a],
                "time": [execution_time],
            }
            list_of_agents = im_instance.get_agents()
            for agent in list_of_agents:
                result_row[agent.name] = [agent.seed]
                result_row[agent.name + '_spread'] = [agent.spread]
            df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
            self.reset_agents(list_of_agents)
        print(df)

    def test_bigdata(self):
        # Test Network
        df = pd.DataFrame()
        g = read_adjacency_matrix('../../data/BigTestData.txt')
        algo = ['celfpp', 'static_greedy']
        dict_of_agents = self.__create_agents__(num_agents=2)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                     diff_model='ic', inf_prob=None, r=100,
                                                     insert_opinion=False, endorsement_policy='random')
            seed = im_instance.run()
            spread = im_instance.result['spread']
            execution_time = im_instance.result['execution_time']
            result_row = {
                "algorithm": [a],
                "time": [execution_time],
            }
            list_of_agents = im_instance.get_agents()
            for agent in list_of_agents:
                result_row[agent.name] = [agent.seed]
                result_row[agent.name + '_spread'] = [agent.spread]
            df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
            self.reset_agents(list_of_agents)
        print(df)