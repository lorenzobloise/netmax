import unittest
import src.single_agent.influence_maximization as im
from src.common.utils import read_graph

"""
This python file contains the tests for all the diffusion model.
So we have already the seeds set for all the agent and run only the model
"""
class DiffusionModelTests(unittest.TestCase):

    def test_independent_cascade(self):
        seed_set = [56, 58, 48, 62, 52, 53, 28, 55]
        graph = read_graph("../../data/network.txt")
        im_instance = im.InfluenceMaximization(graph, 'Agent_0', budget=8, diff_model='ic', inf_prob='uniform', r=100)
        mapped_seed_set = [im_instance.mapping[s] for s in seed_set]
        spread = im.simulation(im_instance.graph, diff_model=im_instance.diff_model, agent_name=im_instance.agent, seed=mapped_seed_set, r=im_instance.r)
