import networkx as nx
from src.common import utils
from src.single_agent.algorithms.algorithm import Algorithm
from src.single_agent.algorithms.simulation_based.simulation_based import SimulationBasedAlgorithm
from src.single_agent.algorithms.proxy_based.proxy_based import ProxyBasedAlgorithm
from src.single_agent.diffusion_models.diffusion_model import DiffusionModel
from src.common.influence_probabilities.influence_probability import InfluenceProbability
import time
from tqdm import tqdm

def activate_node(graph, node, agent_name):
    """
    Activate a node in the graph by setting its status to 'ACTIVE' and the agent name that activated it.
    :param graph: The input graph (networkx.DiGraph).
    :param node: The node to activate.
    :param agent_name: The name of the agent who activates the node.
    """
    graph.nodes[node]['status'] = 'ACTIVE'
    graph.nodes[node]['agent'] = agent_name

def active_nodes(graph):
    return [u for u in graph.nodes if is_active(u, graph)]

def inactive_nodes(graph):
    return [u for u in graph.nodes if not is_active(u, graph)]

def is_active(node, graph):
    return graph.nodes[node]['status'] == 'ACTIVE'

def remove_isolated_nodes(graph):
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes)}
    nx.relabel_nodes(graph, mapping, copy=False)
    return mapping

def simulation(graph, diff_model, agent_name, seed, r=10000, community=None):
    spread = 0
    for _ in tqdm(range(r), desc="Simulations", position=0, leave=True):
        active_set = diff_model.activate(graph, agent_name, seed)
        if community is not None:
            active_set = [node for node in active_set if node in community]
        spread += len(active_set)
    spread /= r
    return spread

def simulation_delta(graph, diff_model, agent, seed1, seed2, r=10000, community=None):
    spread = 0
    for _ in range(r):
        active_set_1 = diff_model.activate(graph, agent, seed1)
        active_set_2 = diff_model.activate(graph, agent, seed2)
        if community is not None:
            active_set_1 = [node for node in active_set_1 if node in community]
            active_set_2 = [node for node in active_set_2 if node in community]
        spread += len([x for x in active_set_1 if x not in active_set_2])
    spread = spread / r
    return spread

class InfluenceMaximization:

    # TODO: fix hierarchy method in utils
    def __init__(self, input_graph: nx.DiGraph, agent: str, budget: int, alg: str = 'celf', diff_model: str = 'ic',
                 inf_prob: str = 'uniform', insert_prob: bool = False, inv_edges: bool = False, r: int = 100):
        """
        Create an instance of the Influence Maximization problem.
        :param input_graph: The input directed graph (networkx.DiGraph).
        :param agent: The name of the agent who wants to maximize his influence.
        :param budget: The budget of the agent expressed in number of nodes initially influenced.
        :key alg: Algorithm that solves the maximization problem. The framework implements different algorithms, default is 'celf'.
        :key diff_model: Diffusion model that models the influence spreading among the nodes. The framework implements different diffusion models, default is 'ic'.
        :key inf_prob: Probability distribution to generate (if needed) the probabilities of influence between nodes. The framework implements different probability distributions, default is 'uniform'.
        :key insert_prob: If True, labels the edges with the influence probability.
        :key inv_edges: If True, inverts the graph edges (edge (u,v) becomes (v,u)).
        :key r: Number of simulations to execute. Default is 10000.
        """
        self.graph = input_graph.copy()
        self.agent = agent
        self.budget = budget
        n_nodes = len(self.graph.nodes)
        if self.budget > n_nodes:
            raise ValueError(f"The budget ({self.budget}) exceeds the number of nodes in the graph ({n_nodes}) by {self.budget - n_nodes}")
        hierarchy: dict = dict(utils.find_hierarchy(Algorithm) +
                               utils.find_hierarchy(DiffusionModel) +
                               utils.find_hierarchy(InfluenceProbability))
        for (k, v) in {'alg': alg, 'diff_model': diff_model, 'inf_prob': inf_prob}.items():
            if v not in hierarchy.keys():
                raise ValueError(f"Argument '{v}' not supported for field '{k}'")
        self.insert_prob = insert_prob
        self.inv_edges = inv_edges
        self.r = r
        self.mapping = None
        self.__preprocess__()
        self.inf_prob = hierarchy[inf_prob]()
        self.diff_model = hierarchy[diff_model]()
        self.diff_model.preprocess_data(self.graph)
        self.alg = hierarchy[alg](graph=self.graph, agent=self.agent, budget=self.budget, diff_model=self.diff_model, r=self.r)
        self.result = None

    def __preprocess__(self):
        self.mapping = remove_isolated_nodes(self.graph)
        if self.inv_edges:
            self.graph = self.graph.reverse(copy=False)
        if self.insert_prob:
            for (source, target) in self.graph.edges:
                self.graph.edges[source, target]['p'] = self.inf_prob.get_probability(self.graph, source, target)
        for node in self.graph.nodes:
            self.graph.nodes[node]['status'] = 'INACTIVE'

    def run(self):
        start_time = time.time()
        seed = self.alg.run()
        spread = simulation(self.graph, self.diff_model, self.agent, seed, self.r)
        execution_time = time.time() - start_time
        inverse_mapping = {new_label: old_label for (old_label, new_label) in self.mapping.items()}
        self.result = {
            'seed': [inverse_mapping[s] for s in seed],
            'spread': spread,
            'execution_time': execution_time
        }
        return self.result['seed']