# Standard Library
from itertools import combinations, product
from math import sqrt


##############
# MATH HELPERS
##############


def dot_product(x, y):
    return sum(x_i * y_i for x_i, y_i in zip(x, y))


def norm(x):
    return sqrt(sum(x_i ** 2 for x_i in x))


def compute_cosine_sim(x, y):
    return dot_product(x, y) / (norm(x) * norm(y))


def compute_euclidean_dist(x, y):
    return sqrt(sum((y_i - x_i) ** 2 for x_i, y_i in zip(x, y)))


##############
# ALGO HELPERS
##############


def create_node_similarity_matrix(adj_matrix, metric):
    num_nodes = len(adj_matrix)
    N = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, src_vector in enumerate(adj_matrix):
        for j, targ_vector in enumerate(adj_matrix):
            if j >= i:
                break

            if metric == "cosine":
                sim = compute_cosine_sim(src_vector, targ_vector)
            elif metric == "euclidean":
                sim = compute_euclidean_dist(src_vector, targ_vector)

            N[i][j] = sim
            N[j][i] = sim

    return N


def find_best_merge(C, metric):
    merge_indices = ()
    if metric == "cosine": # [-1, 1], 1 = similar, -1 = different
        best_sim = -1.0
    elif metric == "euclidean": # [0, inf], 0 = similar, inf = different
        best_sim = float("inf")

    for c_i, neighbors in enumerate(C):
        for c_j, similarity in enumerate(neighbors):
            if c_j >= c_i:
                break

            if metric == "cosine" and sim <= best_sim:
                continue
            elif metric == "euclidean" and sim >= best_sim:
                continue

            merge_indices, best_sim = (c_i, c_j), sim

    return min(merge_indices), max(merge_indices)


def merge_communities(communities, C, N, metric, linkage):
    # Merge the two most similar communities
    c_i, c_j = find_best_merge(C, metric)
    communities[c_i] |= communities[c_j]
    communities.pop(c_j)

    # Update the community similarity matrix, C
    C.pop(c_j)
    for row in C:
        row.pop(c_j)

    for c_j in range(len(C[c_i])):
        if c_j == c_i:
            continue

        sims = []
        for u, v in product(communities[c_i], communities[c_j]):
            sims.append(N[u][v])

        if linkage == "single":
            sim = min(sims)
        elif linkage == "complete":
            sim = max(sims)

        C[c_i][c_j] = sim
        C[c_j][c_i] = sim

    return communities, C


######
# MAIN
######


# TODO: Optimize for modularity
def hierarchical_cluster(adj_matrix, metric="euclidean", linkage="single",
                         size=None):
    """
    """

    metric, linkage = metric.lower(), linkage.lower()

    communities = [{node} for node in range(len(adj_matrix))]
    N = create_node_similarity_matrix(adj_matrix, metric)
    C = N # Community similarity matrix

    while True:
        if size and len(communities) == size:
            break

        communities, C = merge_communities(communities, C, N, metric, linkage)

    return communities
