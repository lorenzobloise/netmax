import unittest
import pandas as pd
from utils import read_adjacency_matrix
from utils import read_weighted_and_signed_adjacency_matrix
from netmax import influence_maximization as im

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class GeneralTests(unittest.TestCase):

    def __reset_agents__(self, agents_dict):
        """
        At the end of every algorithm, reset the seed sets of the agents.
        :param agents_dict: the 'agents' dictionary.
        """
        for agent in agents_dict:
            agent.seed = []

    def test_im_unsigned_network(self):
        df = pd.DataFrame()
        # SET THE PARAMETERS
        # Insert the agents you want in the format <name: budget>
        agents_dict = {
            'Agent_0': 10,
            'Agent_1': 10
        }
        # Un-comment the algorithms you want to execute
        algos = [
            # Simulation-based
            #'mcgreedy',
            #'celf',
            #'celfpp',
            # Proxy-based
            #'outdeg',
            #'degdis',
            #'group_pr',
            # Sketch-based
            'static_greedy',
            'ris',
            'tim',
            'tim_p'
        ]
        # Insert the path of your graph (which has to be in the appropriate format, see utils.py for some useful
        # graph pre-processing)
        g = read_adjacency_matrix('../data/network.txt')
        diff_model = 'ic'
        inf_prob = None
        first_random_seed = False
        num_simulations = 100
        insert_opinion = False
        endorsement_policy = 'random'
        verbose = True
        # Start the influence maximization process with an algorithm at the time
        for a in algos:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=agents_dict, alg=a,
                                                    diff_model=diff_model, inf_prob=inf_prob,
                                                    first_random_seed=first_random_seed, r=num_simulations,
                                                    insert_opinion=insert_opinion, endorsement_policy=endorsement_policy,
                                                    verbose=verbose)
            seed, spread, execution_time = im_instance.run()
            result_row = {
                "algorithm": [a],
                "time": [execution_time],
            }
            agents_list = im_instance.get_agents()
            for agent in agents_list:
                result_row[agent.name] = [agent.seed]
                result_row[agent.name + '_spread'] = [agent.spread]
            df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
            self.__reset_agents__(agents_list)
        print(df)

    def test_im_signed_network(self):
        df = pd.DataFrame()
        # SET THE PARAMETERS
        # Insert the agents you want in the format <name: budget>
        agents_dict = {
            'Agent_0': 10,
            'Agent_1': 10
        }
        # Un-comment the algorithms you want to execute
        algos = [
            # Simulation-based
            #'mcgreedy',
            #'celf',
            #'celfpp',
            # Proxy-based
            #'outdeg',
            'degdis',
            #'group_pr',
            # Sketch-based
            #'static_greedy',
            #'ris',
            #'tim',
            #'tim_p'
        ]
        # If you don't have the weights execute the first method instead of the second
        #g = read_signed_adjacency_matrix('../data/wikiconflict_signed.txt') # First method
        g = read_weighted_and_signed_adjacency_matrix('../data/wikiconflict-signed_edgelist.txt') # Second method
        diff_model = 'sp_f2dlt'
        inf_prob = None
        num_simulations = 10
        insert_opinion = False
        endorsement_policy = 'random'
        verbose = True
        for a in algos:
            im_instance = im.InfluenceMaximization(input_graph=g, agents=agents_dict, alg=a,
                                                    diff_model=diff_model, inf_prob=inf_prob, r=num_simulations,
                                                    insert_opinion=insert_opinion, endorsement_policy=endorsement_policy,
                                                    verbose=verbose)
            seed, spread, execution_time = im_instance.run()
            result_row = {
                "algorithm": [a],
                "time": [execution_time],
            }
            agents_list = im_instance.get_agents()
            for agent in agents_list:
                result_row[agent.name] = [agent.seed]
                result_row[agent.name + '_spread'] = [agent.spread]
            df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
            self.__reset_agents__(agents_list)
        print(df)