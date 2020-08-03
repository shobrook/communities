# Standard Library
from itertools import product, combinations

# Third Party
import numpy as np


######
# MAIN
######


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


def modularity_matrix(A):
    k_i = np.expand_dims(A.sum(axis=1), axis=1)
    k_j = k_i.T
    norm = 1 / k_i.sum()
    K = norm * np.matmul(k_i, k_j)

    return norm * (A - K)


def modularity(M, communities):
    C = np.zeros_like(M)
    for community in communities:
        for i, j in combinations(community, 2):
            C[i, j] = 1.0
            C[j, i] = 1.0

    return np.tril(np.multiply(M, C), 0).sum()