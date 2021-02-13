# Standard Library
import random
from copy import deepcopy

# Third Party
import numpy as np

# Local
from ..utilities import laplacian_matrix


##############
# MATH HELPERS
##############


def eigenvector_matrix(L, n):
    eigvals, eigvecs = np.linalg.eig(L)
    sorted_eigs = sorted(zip(eigvals, eigvecs.T), key=lambda e: e[0])

    n_eigvecs = []
    for index, (eigval, eigvec) in enumerate(sorted_eigs):
        if not index:
            continue
        elif index == n:
            break

        n_eigvecs.append(eigvec)

    return np.vstack(n_eigvecs).T


#################
# K-MEANS HELPERS
#################


def init_communities(num_nodes, k):
    # QUESTION: Forgy method vs. Random Partition method?
    return [{i} for i in random.sample(range(num_nodes), k)]


def calc_centroids(V, communities):
    centroids = []
    for community in communities:
        centroid = V[list(community)].mean(axis=0)
        centroids.append(centroid)

    C = np.vstack(centroids)
    return C


def update_assignments(V, C, communities):
    for i in range(len(V)):
        best_sim, best_comm_index = -1, 0
        for c_i in range(len(C)):
            cosine_sim = np.dot(V[i], C[c_i])
            cosine_sim /= np.linalg.norm(V[i]) * np.linalg.norm(C[c_i])
            if cosine_sim < best_sim:
                continue

            best_sim = cosine_sim
            best_comm_index = c_i

        communities[best_comm_index].add(i)

    return communities


######
# MAIN
######


def spectral_clustering(adj_matrix : np.ndarray, k : int) -> list:
    L = laplacian_matrix(adj_matrix)
    V = eigenvector_matrix(L, k)

    communities = init_communities(len(adj_matrix), k)
    while True:
        C = calc_centroids(V, communities)
        updated_communities = update_assignments(V, C, deepcopy(communities))

        if updated_communities == communities:
            break

        communities = updated_communities

    return communities
