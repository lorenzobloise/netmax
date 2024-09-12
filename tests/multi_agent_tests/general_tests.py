import unittest
import random
import pandas as pd
from common.utils import read_adjacency_matrix
import multi_agent.competitive_influence_maximization as cim
from multi_agent.agent import Agent

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


class GeneralTests(unittest.TestCase):

    def __create_agents__(self, num_agents):
        agents = []
        for i in range(num_agents):
            agent_name = 'Agent_' + str(i)
            agent = Agent(agent_name, random.randint(10, 10))
            agent.__setattr__('id', i)
            agents.append(agent)
        return agents

    def reset_agents(self, agents):
        for agent in agents:
            agent.seed = []
        return agents

    def test_network(self):
        # Test Network
        df = pd.DataFrame()
        g = read_adjacency_matrix('../../data/network.txt')
        algo = ['mcgreedy', 'celf', 'celfpp']
        list_of_agents = self.__create_agents__(num_agents=4)
        for a in algo:
            cim_instance = cim.CompetitiveInfluenceMaximization(input_graph=g, agents=list_of_agents, alg=a,
                                                                diff_model='ic', inf_prob=None, r=100,
                                                                insert_opinion=False, endorsement_policy='random')
            seed = cim_instance.run()
            spread = cim_instance.result['spread']
            execution_time = cim_instance.result['execution_time']
            result_row = {
                "algorithm": [a],
                "time": [execution_time],
            }
            for agent in list_of_agents:
                result_row[agent.name] = [agent.seed]
                result_row[agent.name + '_spread'] = [agent.spread]
            df = pd.concat([df, pd.DataFrame(result_row)], ignore_index=True)
            list_of_agents = self.reset_agents(list_of_agents)
        print(df)
