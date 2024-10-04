import unittest
import random
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
        g = read_adjacency_matrix('../data/network.txt')
        algo = ['mcgreedy', 'celf', 'celfpp', 'degdis', 'group_pr', 'outdeg', 'tim', 'ris', 'tim_p', 'static_greedy']
        dict_of_agents = self.__create_agents__(num_agents=2)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                     diff_model='tr', inf_prob='opinion', r=1,
                                                     insert_opinion=True, endorsement_policy='random', verbose=True)
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
        #g = read_signed_adjacency_matrix('../data/wiki-elec.txt')
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt')
        algo = ['group_pr']
        dict_of_agents = self.__create_agents__(num_agents=2)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                    diff_model='sp_f2dlt', inf_prob=None, r=1, insert_opinion=False,
                                                    endorsement_policy='random', verbose=True)
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
        g = read_weighted_and_signed_adjacency_matrix('../data/wiki-elec_edgelist.txt')
        dict_of_agents = self.__create_agents__(num_agents=2)
        im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg='tim_p',
                                                    diff_model='sp_f2dlt', inf_prob=None, r=1, insert_opinion=False,
                                                    endorsement_policy='random', verbose=True)
        im_instance.agents[0].seed = [27, 79, 3124, 6100, 68, 345, 6759, 4359, 5977, 1328]
        im_instance.agents[1].seed = [173, 2872, 5087, 4, 315, 6447, 867, 1095, 1306, 1343]
        active_sets = im_instance.diff_model.activate(im_instance.graph, im_instance.agents)
        print(len(active_sets['Agent_0']))
        print(len(active_sets['Agent_1']))