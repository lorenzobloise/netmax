import networkx as nx

def read_adjacency_matrix(input_path):
    """
    :return: The directed graph in networkx format
    :param input_path: Path of the dataset (.txt file)
    """
    graph = nx.DiGraph()
    with open(input_path, 'r') as f:
        data_lines = f.readlines()
    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), p=float(weight))
    return graph

def process_graph_file(file_path):
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

def __binomial_coefficient__(n, k):
    C = [[-1 for _ in range(k+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(min(i, k+1)):
            # Base cases
            if j == 0 or j == i:
                C[i][j] = 1
            # Calculate value using previously stored values
            else:
                C[i][j] = C[i-1][j-1] + C[i-1][j]
    return C[n][k]

# TODO
def read_gml(input_path):
    return None

def __my_similarity__(list1, list2):
    number_of_elements_of_list1_in_list2 = 0
    for x in list1:
        if x in list2:
            number_of_elements_of_list1_in_list2 += 1
    return number_of_elements_of_list1_in_list2 / len(list1)

def __calculate_similarities__(dataframe):
    # Calculate similarities between seed sets of different algorithms
    algorithms = dataframe["algorithm"].unique()
    for i, row in dataframe.iterrows():
        current_seed_set = row["seed"]
        for elem in algorithms:
            other_seed_set=dataframe[dataframe["algorithm"] == elem].iloc[0]["seed"]
            dataframe.loc[i, f'similarity_{elem}'] = __my_similarity__(current_seed_set, other_seed_set)
    return dataframe

def find_hierarchy(superclass):
    subclasses = []
    for subclass in superclass.__subclasses__():
        if hasattr(subclass, 'name'):
            subclasses.append((subclass.name, subclass))
        else:
            subclasses.extend(find_hierarchy(subclass))
    return subclasses