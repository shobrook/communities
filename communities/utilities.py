# Standard Library
from itertools import product, combinations
from typing import Callable

# Third Party
import numpy as np


######
# MAIN
######


def intercommunity_matrix(adj_matrix : np.ndarray, communities : list,
                          aggr : Callable = sum) -> np.ndarray:
    num_nodes = len(communities)
    intercomm_adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, src_comm in enumerate(communities):
        for j, targ_comm in enumerate(communities):
            if j > i:
                break

            edge_weights = []
            for u, v in product(src_comm, targ_comm):
                edge_weights.append(adj_matrix[u, v])

            edge_weight = aggr(edge_weights)
            intercomm_adj_matrix[i, j] = edge_weight
            intercomm_adj_matrix[j, i] = edge_weight

    return intercomm_adj_matrix


def laplacian_matrix(adj_matrix : np.ndarray) -> np.ndarray:
    diagonal = adj_matrix.sum(axis=0)
    D = np.diag(diagonal)
    L = D - adj_matrix

    return L


def modularity_matrix(adj_matrix : np.ndarray) -> np.ndarray:
    k_i = np.expand_dims(adj_matrix.sum(axis=1), axis=1)
    k_j = k_i.T
    norm = 1 / k_i.sum()
    K = norm * np.matmul(k_i, k_j)

    return norm * (adj_matrix - K)


def modularity(mod_matrix : np.ndarray, communities : list) -> float:
    C = np.zeros_like(mod_matrix)
    for community in communities:
        for i, j in combinations(community, 2):
            C[i, j] = 1.0
            C[j, i] = 1.0

    return np.tril(np.multiply(mod_matrix, C), 0).sum()
