import unittest
import random
import pandas as pd
from utils import read_adjacency_matrix
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

    def __reset_agents__(self, agents):
        for agent in agents:
            agent.seed = []

    def test(self):
        df = pd.DataFrame()
        g = read_adjacency_matrix('../data/network.txt')
        algo = ['celf', 'celfpp']
        dict_of_agents = self.__create_agents__(num_agents=2)
        for a in algo:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg=a,
                                                     diff_model='ic', inf_prob=None, first_random_seed=True, r=100,
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
            self.__reset_agents__(list_of_agents)
        print(df)

    def test_signed_graph(self):
        df = pd.DataFrame()
        #g = read_signed_adjacency_matrix('../data/wikiconflict_signed.txt')
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt')
        algo = ['degdis']
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
            self.__reset_agents__(list_of_agents)
        print(df)
        """
        print("---------------------")
        history = im_instance.diff_model.get_history()
        new_history = {}
        for it in history:
            (active_sets, pending_nodes, quiescent_nodes) = history[it]
            #new_history[it] = (sum([len(a) for a in active_sets.values()]), len(pending_nodes), sum([len(a) for a in quiescent_nodes.values()]))
            new_history[it] = ([im_instance.inverse_mapping[x] for x in [a[i] for a in active_sets.values() for i in range(len(a))]],
                               [im_instance.inverse_mapping[x] for x in pending_nodes.keys()],
                               [im_instance.inverse_mapping[x] for x in [q[i] for q in quiescent_nodes.values() for i in range(len(q))]])
        print(new_history)
        """

    def test_diffusion_model(self):
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt')
        dict_of_agents = self.__create_agents__(num_agents=1)
        im_instance = im.InfluenceMaximization(input_graph=g, agents=dict_of_agents, alg='tim_p',
                                                    diff_model='sp_f2dlt', inf_prob=None, r=1, insert_opinion=False,
                                                    endorsement_policy='random', verbose=True)
        im_instance.agents[0].seed = [im_instance.mapping[x] for x in [64]]
        active_sets = im_instance.diff_model.activate(im_instance.graph, im_instance.agents)
        print(len(active_sets['Agent_0']))
        #print(len(active_sets['Agent_1']))
        print("---------------------")
        history = im_instance.diff_model.get_history()
        new_history = {}
        for it in history:
            (active_sets, pending_nodes, quiescent_nodes) = history[it]
            new_history[it] = (sum([len(a) for a in active_sets.values()]), len(pending_nodes), sum([len(a) for a in quiescent_nodes.values()]))
        print(new_history)