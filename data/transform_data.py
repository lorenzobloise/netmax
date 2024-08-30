import networkx as nx
import utils

def transform_edge_list(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            values = line.strip().split()
            if len(values) == 3:
                values[2] = f"{{'p': {values[2]}}}"
                outfile.write(' '.join(values)+'\n')


def read_graph(input_path):
    g = nx.read_edgelist(input_path,create_using=nx.DiGraph,nodetype=int)
    nodes = [i for i in range(len(g.nodes))]
    g_tmp = nx.DiGraph()
    g_tmp.add_nodes_from(nodes)
    g_tmp.add_edges_from(g.edges(data=True))
    return g_tmp

def transform_graph(g):
    for (n, attr) in g.nodes(data=True):
        attr['status'] = utils.NodeStatus.INACTIVE.value
        attr['agent'] = utils.NO_AGENT_LABEL


def write_graph(g, output_path):
    nx.write_gml(g,output_path)
    return

'''
read network file into a graph
network format:
    node_num edge_num
    start end weight
:param input_path: the file path of network
'''
def read_dataset(input_path):
    graph = nx.DiGraph()
    data_lines = open(input_path, 'r').readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])
    nodes = [i for i in range(node_num)]
    graph.add_nodes_from(nodes)
    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), p=float(weight))
    return graph