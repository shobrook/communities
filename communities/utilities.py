# Standard Library
from itertools import product


######
# MAIN
######


def is_left_triangular(adj_matrix):
    for i, neighbors in enumerate(adj_matrix):
        if len(neighbors) > i:
            return False
    
    return True


def symmetrize_matrix(adj_matrix):
    if not is_left_triangular(adj_matrix):
        return adj_matrix
    
    num_nodes = len(adj_matrix)
    sym_adj_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, neighbors in enumerate(adj_matrix):
        for j, weight in enumerate(neighbors):
            sym_adj_matrix[i][j] = weight
            sym_adj_matrix[j][i] = weight
    
    return sym_adj_matrix


def binarize_matrix(adj_matrix, threshold):
    num_nodes = len(adj_matrix)
    bin_adj_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, neighbors in enumerate(adj_matrix):
        for j, weight in enumerate(neighbors):
            bin_adj_matrix[i][j] = 1.0 if abs(weight) >= threshold else 0.0

    return bin_adj_matrix


def create_intercommunity_graph(adj_matrix, communities, aggr=sum):
    num_nodes = len(communities)
    intercomm_adj_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, src_comm in enumerate(communities):
        for j, targ_comm in enumerate(communities):
            if j > i:
                break
            
            edge_weights = []
            for u, v in product(src_comm, targ_comm):
                edge_weights.append(adj_matrix[u][v])
            
            edge_weight = aggr(edge_weights)
            intercomm_adj_matrix[i][j] = edge_weight
            intercomm_adj_matrix[j][i] = edge_weight
    
    return intercomm_adj_matrix
