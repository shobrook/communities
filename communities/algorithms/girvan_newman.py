# Third Party
import networkx as nx
import numpy as np


# TODO: Rewrite this in pure Python


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


# TODO: Replace with function used in louvain.py
def get_modularity(G):
    B = nx.modularity_matrix(G, weight="weight")
    comps = list(nx.connected_components(G))

    m, sigma = 0.0, 0.0
    for i, j, data in G.edges(data=True):
        m += data["weight"]
        if any({i, j}.issubset(comp) for comp in comps):
            sigma += B[i, j]
    
    return (1 / (2 * m)) * sigma


######
# MAIN
######


def girvan_newman(adj_matrix):
    G = nx.from_numpy_matrix(np.array(adj_matrix))
    G.remove_edges_from(nx.selfloop_edges(G))

    best_Q, best_G = 0.0, G.copy()
    while True:
        prune_edges(G)
        Q = get_modularity(G)

        if Q >= best_Q:
            best_Q, best_G = Q, G.copy()
        else:
            break
    
    return list(nx.connected_components(best_G))
