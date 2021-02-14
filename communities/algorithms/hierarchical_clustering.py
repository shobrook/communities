# Standard Library
from itertools import product
from math import sqrt
from statistics import mean

# Third Party
import numpy as np

# Local
from ..utilities import modularity_matrix, modularity


##############
# MATH HELPERS
##############


def inverse_euclidean_dist(A):
    p1 = np.sum(A ** 2, axis=1)[:, np.newaxis]
    p2 = -2 * np.dot(A, A)
    p3 = np.sum(A.T ** 2, axis=1)
    E = 1 / np.sqrt(p1 + p2 + p3)

    return E


def cosine_sim(A):
    d = A @ A.T
    norm = (A * A).sum(0, keepdims=True) ** 0.5
    C = d / norm / norm.T

    return C


##############
# ALGO HELPERS
##############


# TODO: Implement Chebyshev and Manhattan distances
def node_similarity_matrix(adj_matrix, metric):
    if metric == "cosine":
        N = cosine_sim(adj_matrix)
    elif metric == "euclidean":
        N = inverse_euclidean_dist(adj_matrix)

    np.fill_diagonal(N, 0.0)
    return N


def find_best_merge(C):
    merge_indices = np.unravel_index(C.argmax(), C.shape)
    return min(merge_indices), max(merge_indices)


def merge_communities(communities, C, N, linkage):
    # Merge the two most similar communities
    c_i, c_j = find_best_merge(C)
    communities[c_i] |= communities[c_j]
    communities.pop(c_j)

    # Update the community similarity matrix, C
    C = np.delete(C, c_j, axis=0)
    C = np.delete(C, c_j, axis=1)

    for c_j in range(len(C)):
        if c_j == c_i:
            continue

        sims = []
        for u, v in product(communities[c_i], communities[c_j]):
            sims.append(N[u, v])

        if linkage == "single":
            similarity = min(sims)
        elif linkage == "complete":
            similarity = max(sims)
        elif linkage == "mean":
            similarity = mean(sims)
        # TODO: Add centroid-linkage

        C[c_i, c_j] = similarity
        C[c_j, c_i] = similarity

    return communities, C


######
# MAIN
######


def hierarchical_clustering(adj_matrix : np.ndarray, metric : str = "cosine",
                            linkage : str = "single", n : int = None) -> list:
    metric, linkage = metric.lower(), linkage.lower()

    communities = [{node} for node in range(len(adj_matrix))]
    M = modularity_matrix(adj_matrix)
    N = node_similarity_matrix(adj_matrix, metric)
    C = np.copy(N) # Community similarity matrix

    best_Q = -0.5
    while True:
        Q = 0.0
        if not n:
            Q = modularity(M, communities)
            if Q <= best_Q:
                break
        elif n and len(communities) == n:
            break

        communities, C = merge_communities(communities, C, N, linkage)
        best_Q = Q

    return communities
