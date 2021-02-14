# Third Party
import networkx as nx
import numpy as np

# Local
from ..utilities import modularity_matrix, modularity


#########
# HELPERS
#########


def prune_edges(G):
    init_num_comps = nx.number_connected_components(G)
    curr_num_comps = init_num_comps

    # TODO: Recalculate betweenness of only the edges affected by the removal
    while curr_num_comps <= init_num_comps:
        bw_centralities = nx.edge_betweenness_centrality(G, weight="weight")
        bw_centralities = sorted(
            bw_centralities.items(),
            key=lambda e: e[1],
            reverse=True
        )

        max_bw = None
        for edge, bw in bw_centralities:
            if max_bw is None:
                max_bw = bw

            if max_bw == bw:
                G.remove_edge(*edge)
            else:
                break

        curr_num_comps = nx.number_connected_components(G)

    return G


def animation_data(A, P_history, Q_history):
    num_nodes = len(A)
    frames = []
    for P, Q in zip(P_history, Q_history):
        _P = [0 for _ in range(num_nodes)]
        for index, partition in enumerate(P):
            for node in partition:
                _P[node] = index

        frames.append({"C": _P, "Q": Q})

    return frames



######
# MAIN
######


def girvan_newman(adj_matrix : np.ndarray, n : int = None) -> list:
    M = modularity_matrix(adj_matrix)
    G = nx.from_numpy_matrix(adj_matrix)
    num_nodes = G.number_of_nodes()
    G.remove_edges_from(nx.selfloop_edges(G))

    best_P = list(nx.connected_components(G)) # Partition
    best_Q = modularity(M, best_P)
    P_history = [best_P]
    Q_history = [best_Q]
    while True:
        last_P = P_history[-1]
        if not n and len(last_P) == num_nodes:
            return best_P, animation_data(adj_matrix, P_history, Q_history) # TODO: Only up to index of best_P
        elif n and len(last_P) == n:
            return last_P, animation_data(adj_matrix, P_history, Q_history)

        G = prune_edges(G)
        P = list(nx.connected_components(G))
        Q = modularity(M, P)
        if Q >= best_Q:
            best_Q = Q
            best_P = P

        P_history.append(P)
        Q_history.append(Q)
