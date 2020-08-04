# Third Party
import networkx as nx
import numpy as np

# Local
from ..utilities import modularity_matrix, modularity

# TODO: Get rid of NetworkX as a dependency


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


######
# MAIN
######


def girvan_newman(adj_matrix, n=None):
    M = modularity_matrix(adj_matrix)
    G = nx.from_numpy_matrix(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    communities = list(nx.connected_components(G))
    
    best_Q = -0.5
    while True:
        Q = 0.0
        if not n:
            Q = modularity(M, communities)
            if Q <= best_Q:
                break
        elif n and len(communities) == n:
            break

        G = prune_edges(G)
        communities = list(nx.connected_components(G))
        best_Q = Q
    
    return communities