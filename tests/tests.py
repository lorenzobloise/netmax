import unittest
import random

import networkx as nx
import pandas as pd
from utils import read_adjacency_matrix
from utils import read_signed_adjacency_matrix
from utils import read_weighted_and_signed_adjacency_matrix
import influence_maximization as im

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

    def test(self):
        df = pd.DataFrame()
        g = read_adjacency_matrix('../data/Instagram70k.txt')
        algo = ['static_greedy']
        dict_of_agents = self.__create_agents__(num_agents=1)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                     diff_model='ic', inf_prob=None, r=1,
                                                     insert_opinion=False, endorsement_policy='random', verbose=True)
            seed, spread, execution_time = im_instance.run()
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

    def test_signed_graph(self):
        df = pd.DataFrame()
        #g = read_signed_adjacency_matrix('../data/wikiconflict-signed.txt')
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt')
        algo = ['degdis']
        dict_of_agents = self.__create_agents__(num_agents=2)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                    diff_model='sp_f2dlt', inf_prob=None, r=1, insert_opinion=False,
                                                    endorsement_policy='random', verbose=False)
            seed, spread, execution_time = im_instance.run()
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

    def test_diffusion_model(self):
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt')
        dict_of_agents = self.__create_agents__(num_agents=2)
        im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg='tim_p',
                                                    diff_model='sp_f2dlt', inf_prob=None, r=1, insert_opinion=False,
                                                    endorsement_policy='random', verbose=True)
        im_instance.agents[0].seed = [115, 125, 100, 64, 591, 76, 624, 595, 66, 19]
        im_instance.agents[1].seed = [37, 506, 435, 28, 109, 86, 27, 12, 80, 4]
        for i in range(100):
            active_sets = im_instance.diff_model.activate(im_instance.graph, im_instance.agents)
            print(len(active_sets['Agent_0']))
            print(len(active_sets['Agent_1']))