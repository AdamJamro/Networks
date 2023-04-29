import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# G = nx.barabasi_albert_graph(100, 2)


average_packet_size = 1000 * 8  # 1000 B


def create_graph():
    # NETWORK STRUCTURE INIT
    df = pd.DataFrame(
        {'from': [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13,
                  13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 1, 9, 3],
         'to': [2, 3, 5, 8, 9, 3, 7, 9, 4, 9, 6, 8, 9, 8, 9, 11, 14, 15, 16, 16, 17, 12, 13, 18, 17, 18, 19,
                14, 19, 15, 16, 0, 19, 0, 17, 0, 18, 0, 19, 0, 16, 19, 12]})
    G = nx.from_pandas_edgelist(df, source='from', target='to')
    for edge in G.edges:
        G[edge[0]][edge[1]]['weight'] = 0
        G[edge[0]][edge[1]]['capacity'] = \
            (800 / np.log(G.degree(edge[0]) + G.degree(edge[1])) + 170) * average_packet_size
    # weight is being measured in # of packets
    # capacity is measured in bits

    G[1][16]['capacity'] *= 3 / 2  # bonus for being crucial part of network
    G[9][19]['capacity'] *= 3 / 2
    G[3][12]['capacity'] *= 3 / 2
    # 3/2 ideally provides us with some sustainability guarantee
    # in theory it should represent expectation that communication between
    # the two of network's main node clusters will remain viable even if
    # one of the three edges enabling the transmission becomes temporarily unavailable

    # this represents the bare minimum amount of bits we could expect for these edges to attain
    G[1][16]['capacity'] += 11 * 2 * average_packet_size  # be generous take max of 11 and 9 (#nodes in clusters)
    G[9][19]['capacity'] += 11 * 2 * average_packet_size
    G[3][12]['capacity'] += 11 * 2 * average_packet_size

    # plt.figure()
    # plt.hist([v for k, v in nx.degree(G)])
    # nx.draw_spring(G, with_labels=True, font_weight='bold')
    # plt.show()
    return G


def calculate_network_infallibility(network, edge_reliability, average_packet_size,
                                    max_latency, intensity_matrix, num_of_trials=500):

    result_probability = 0
    for _ in range(num_of_trials):
        G = nx.Graph(network)

        # # SIMULATE RANDOM NETWORK DISJOINTS
        edges = nx.edges(G)
        for edge in edges:
            if np.random.rand() > edge_reliability:
                G.remove_edge(*edge[:2])  # '*' unpacks an edge tuple

        if not nx.is_connected(G):
            print(f"FAIL. NETWORK IS DISJOINT.")
            _ -= 1
            continue

        # SIMULATE NETWORK COMMUNICATION
        for a in range(intensity_matrix.shape[0]):
            for b in range(a + 1, intensity_matrix.shape[1]):
                temp_path = nx.dijkstra_path(G, a, b)
                for i in range(len(temp_path) - 1):
                    G[temp_path[i]][temp_path[i + 1]]['weight'] += 2 * intensity_matrix[a][b]
                    # weight is the traffic on an edge
                # we're traversing only paths from lower to higher indexes thus omitting half of all paths
                # print(f'intensity_matrix[{a}][{b}]= {intensity_matrix[a][b]}')
                # print(temp_path)

        # IF TRAFFIC IS GREATER THAN CAPACITY THEN BREAK
        failure = 0
        for edge in edges:
            if G[edge[0]][edge[1]]['weight'] > G[edge[0]][edge[1]]['capacity'] / average_packet_size:
                failure = 1
                print(f'''edge[{edge[0]}][{edge[1]}].weight = {G[edge[0]][edge[1]]['weight']}''')
                print(f'''edge[{edge[0]}][{edge[1]}].capacity = {G[edge[0]][edge[1]]['capacity'] / average_packet_size}''')
        if failure == 1:
            _ -= 1
            continue

        # print(f'num of nodes {G.number_of_nodes()}')
        # print(f'num of edges {G.number_of_edges()}')


        total_packet_flow_demand = 0
        for i in range(intensity_matrix.shape[0]):
            for j in range(intensity_matrix.shape[1]):
                total_packet_flow_demand += intensity_matrix[i][j]

        # CALCULATE AVERAGE LATENCY
        latency = 0
        for edge in edges:
            packet_flow_demand = G[edge[0]][edge[1]]['weight']
            packet_flow_capacity = G[edge[0]][edge[1]]['capacity']
            latency += packet_flow_demand / ((packet_flow_capacity / average_packet_size) - packet_flow_demand)
        latency = latency / total_packet_flow_demand
        # print(latency)
        if latency < max_latency:
            result_probability += 1

    return result_probability / num_of_trials


# nodes = nx.nodes(G)
# all_node_pairs = [(a, b) for a in nodes for b in nodes]
# for pair_of_nodes in all_node_pairs:
#     if nx.has_path(G, *pair_of_nodes):
#         print(nx.dijkstra_path(G, *pair_of_nodes))

# print([edge.weight for edge in nx.edges(G)])


# INPUT
edge_reliability = 0.999
N = np.ones((20, 20), dtype=int)
N = N - np.diag(np.diag(N))
T_val = 0.0053
trials = 1000
G = create_graph()

# CONST TOPOLOGY, INTENSITY MATRIX VARIES
print("CONST TOPOLOGY, INTENSITY MATRIX VARIES")
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

N = N * 2
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

N = N / 2 * 3
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

N = N / 3 * 4
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

N = N / 4 * 4.02
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

N = N / 4.02 * 4.03
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')


# NEW INPUT
# # in/out total of 2 * 56 packets per second from node 0
N = np.array([[0, 0, 0, 1, 2, 0, 0, 1, 4, 10,  1, 2, 10, 3, 5, 10, 1, 1, 3, 2],

              [0, 0, 2, 2, 3, 2, 1, 1, 1, 1,   0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
              [0, 0, 0, 3, 3, 3, 1, 2, 3, 4,   1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 3, 2, 2, 2,   0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 3, 2, 5, 4,   0, 0, 0, 0, 0, 0, 1, 1, 2, 2],
              [0, 0, 0, 0, 0, 0, 1, 2, 2, 2,   0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 3, 0, 2,   0, 0, 0, 0, 9, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 3, 3,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   0, 0, 0, 0, 1, 2, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 1, 1, 1, 0, 1, 0, 0],

              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 5, 4, 3, 2, 5, 4, 3, 2, 4],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 1, 1, 7, 4, 3, 1, 1, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 1, 2, 3, 1, 2, 3, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 7, 5, 4, 4, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 1, 1, 1, 3],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 1, 3, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              ])

for i in range(N.shape[0]):
    for j in range(i+1, N.shape[1]):
        N[j][i] = N[i][j]

T_val = 0.00424713
edge_reliability = 0.9985
G = create_graph()

# CONST INTENSITY MATRIX, CAPACITY VARIES
print("CONST INTENSITY MATRIX, CAPACITY VARIES")
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')


for edge in nx.edges(G):
    G[edge[0]][edge[1]]['capacity'] *= 1.05

G[1][16]['capacity'] *= 1.1
G[9][19]['capacity'] *= 1.1
G[3][12]['capacity'] *= 1.1

infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

for edge in nx.edges(G):
    G[edge[0]][edge[1]]['capacity'] *= 1.05

G[1][16]['capacity'] *= 1.015
G[9][19]['capacity'] *= 1.015
G[3][12]['capacity'] *= 1.015

infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')


for edge in nx.edges(G):
    G[edge[0]][edge[1]]['capacity'] *= 1.05

G[1][16]['capacity'] *= 1.02
G[9][19]['capacity'] *= 1.02
G[3][12]['capacity'] *= 1.02

infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

for edge in nx.edges(G):
    G[edge[0]][edge[1]]['capacity'] *= 1.02

G[1][16]['capacity'] *= 1.01
G[9][19]['capacity'] *= 1.01
G[3][12]['capacity'] *= 1.01

infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

# CONST INTENSITY MATRIX, TOPOLOGY VARIES
print("CONST INTENSITY MATRIX, TOPOLOGY VARIES")
G = create_graph()
print("original graph:")
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')

mean_capacity = 0
for edge in nx.edges(G):
    mean_capacity += G[edge[0]][edge[1]]['capacity']
mean_capacity = mean_capacity / nx.number_of_edges(G)

print("add an (0,1) edge to relieve between clusters communication:")
G.add_edge(0, 1)
G[0][1]['weight'] = 0
G[0][1]['capacity'] = mean_capacity
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')
print("add an (i,i+2) for each i in [1,7] to enhance the looser cluster's communication:")
for i in range(1, 7):
    if not G.has_edge(i, i+2):
        G.add_edge(i, i+2)
        G[i][i+2]['weight'] = 0
        G[i][i+2]['capacity'] = mean_capacity
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')


print("add an (i, i+9) for each i in [1,9] to merge two separated clusters:")
for i in range(1, 10):
    if not G.has_edge(i, i+9):
        G.add_edge(i, i+9)
        G[i][i+9]['weight'] = 0
        G[i][i+9]['capacity'] = mean_capacity
infallibility = calculate_network_infallibility(network=G,
                                                max_latency=T_val,
                                                edge_reliability=edge_reliability,
                                                average_packet_size=average_packet_size,
                                                intensity_matrix=N,
                                                num_of_trials=trials)
print(f'P[T<T_max] = {infallibility}')
