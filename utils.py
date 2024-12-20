import networkx as nx
import numpy as np
from tqdm import tqdm

def read_adjacency_matrix(input_path):
    """
    :return: The directed graph in networkx format
    :param input_path: Path of the dataset file of the form:
    <from_node> <to_node> <influence_probability>
    """
    graph = nx.DiGraph()
    with open(input_path, 'r') as f:
        data_lines = f.readlines()
    for data_line in tqdm(data_lines[1:], desc="Reading adjacency matrix"):
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), p=float(weight))
    return graph

def read_weighted_and_signed_adjacency_matrix(input_path):
    """
    :return: The directed graph in networkx format
    :param input_path: Path of the dataset file of the form:
    <from_node> <to_node> <influence_probability> <sign>
    """
    graph = nx.DiGraph()
    with open(input_path, 'r') as f:
        data_lines = f.readlines()
    for data_line in tqdm(data_lines[1:], desc="Reading adjacency matrix"):
        start, end, p, sign = data_line.split()
        graph.add_edge(int(start), int(end), p=float(p),s=int(sign))
    return graph

def read_signed_adjacency_matrix(input_path):
    """
    This function reads a signed adjacency matrix of a signed network. The influence probabilities are sampled by
    a binomial distribution with parameters n=in_neighbors_positive, p=number_of_positive_edges/number_of_edges
    :return: The directed graph in networkx format
    :param input_path: Path of the dataset file of the form:
    <from_node> <to_node> <sign>
    """
    graph = nx.DiGraph()
    with open(input_path, 'r') as f:
        data_lines = f.readlines()
    for data_line in tqdm(data_lines[1:], desc="Reading signed adjacency matrix"):
        start, end, sign = data_line.split()
        graph.add_edge(int(start), int(end), s=int(sign))
    # Sample influence probabilities
    # Parameters of the binomial distribution
    # p: probability of success
    # For all nodes, trusted and not trusted in-neighbors, to avoid re-computation
    p_success = len([(u, v) for (u, v, attr) in graph.edges(data=True) if attr['s'] == 1]) / len(graph.edges)
    #p_success= 0.6
    trusted_in_neighbors = {}
    not_trusted_in_neighbors = {}
    num_samples = 100
    for (u,v,attr) in tqdm(graph.edges(data=True), desc="Sampling influence probabilities"):
        # n: number of trusted in-neighbors
        n_trusted = trusted_in_neighbors.get(v,-1)
        n_not_trusted = not_trusted_in_neighbors.get(v,-1)
        # If the lists are not already computed, compute them
        if n_trusted == -1:
            n_trusted = len([x for x in graph.predecessors(v) if graph.edges[x,v]['s'] == 1])
            trusted_in_neighbors[v] = n_trusted
        if n_not_trusted == -1:
            n_not_trusted = len([x for x in graph.predecessors(v) if graph.edges[x,v]['s'] == -1])
            not_trusted_in_neighbors[v] = n_not_trusted
        # Sample the probabilities num_samples times and compute the average
        sum_of_prob = 0
        n = n_trusted if attr['s'] == 1 else n_not_trusted
        for experiment in range(num_samples):
            sum_of_prob += (np.random.binomial(n, p_success)/n) * attr['s']
        influence_probability = sum_of_prob/num_samples
        graph.edges[u,v]['p'] = influence_probability
    write_graph_to_file(graph, input_path[:-4] + "_edgelist.txt")
    return graph

def write_graph_to_file(graph, output_path):
    """
    Writes the edges of the graph to a text file with each row containing:
    fromNode toNode <value of attribute p> <value of attribute sign>
    :param graph: The directed graph in networkx format
    :param output_path: Path of the output file
    """
    with open(output_path, 'w') as f:
        f.write(f"{len(graph.nodes)} {len(graph.edges)}\n")
        for u, v, attr in graph.edges(data=True):
            p = attr.get('p', 0)  # Default to 0 if 'p' is not present
            sign = attr.get('s', 0)  # Default to 0 if 's' is not present
            f.write(f"{u} {v} {p} {sign}\n")

def __my_similarity__(list1, list2):
    """
    Computes the similarity between two seed sets of the same length) by counting the common elements normalized by the seed set length.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    number_of_elements_of_list1_in_list2 = 0
    for x in list1:
        if x in list2:
            number_of_elements_of_list1_in_list2 += 1
    return number_of_elements_of_list1_in_list2 / len(list1)

def __compute_similarities__(dataframe):
    """
    Computes similarities between seed sets found by different algorithms.
    :param: dataframe: A pandas dataframe which contains the results returned by different algorithms.
    :return: The updated dataframe with the similarity values for every couple of results.
    """
    algorithms = dataframe["algorithm"].unique()
    for i, row in dataframe.iterrows():
        current_seed_set = row["seed"]
        for elem in algorithms:
            other_seed_set=dataframe[dataframe["algorithm"] == elem].iloc[0]["seed"]
            dataframe.loc[i, f'similarity_{elem}'] = __my_similarity__(current_seed_set, other_seed_set)
    return dataframe

# Different processing methods, based on the input file format

def __process_1__(file_path):
    """
    Initial format:
    <from_node> <to_node>
    Final format:
    <num_nodes> <num_edges>
    <from_node> <to_node>
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    nodes = set()
    num_edges = len(lines)
    for line in lines:
        node1, node2 = map(int, line.split())
        nodes.add(node1)
        nodes.add(node2)
    num_nodes = len(nodes)
    new_first_line = f"{num_nodes} {num_edges}\n"
    with open(file_path, 'w') as f:
        f.write(new_first_line)
        f.writelines(lines)

def __process_2__(file_path):
    """
    Initial format:
    <from_node>,<to_node>,<sign>
    Final format:
    <num_nodes> <num_edges>
    <from_node> <to_node> <sign>
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    nodes = set()
    num_edges = len(lines)
    for line in lines:
        node1, node2, _ = map(int, line.split(','))
        nodes.add(node1)
        nodes.add(node2)
    num_nodes = len(nodes)
    new_first_line = f"{num_nodes} {num_edges}\n"
    with open(file_path, 'w') as f:
        f.write(new_first_line)
        for line in lines:
            node1, node2, sign = map(int, line.split(','))
            f.write(f"{node1} {node2} {sign}\n")

def __process_3__(file_path):
    """
    Initial format:
    <from_node> <to_node> <sign> <other_term>
    Final format:
    <num_nodes> <num_edges>
    <from_node> <to_node> <sign>
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    nodes = set()
    num_edges = len(lines)
    for line in lines:
        node1, node2, _, _ = map(int, line.split())
        nodes.add(node1)
        nodes.add(node2)
    num_nodes = len(nodes)
    new_first_line = f"{num_nodes} {num_edges}\n"
    with open(file_path, 'w') as f:
        f.write(new_first_line)
        for line in lines:
            node1, node2, sign, _ = map(int, line.split())
            f.write(f"{node1} {node2} {sign}\n")