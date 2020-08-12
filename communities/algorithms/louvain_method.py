# Standard Library
from itertools import combinations, chain
from collections import defaultdict

# Third Party
import numpy as np

# Local
from ..utilities import modularity_matrix, modularity


#########
# HELPERS
#########


def initialize_node_to_comm(adj_matrix):
    return list(range(len(adj_matrix)))


def invert_node_to_comm(node_to_comm):
    communities = defaultdict(set)
    for node, community in enumerate(node_to_comm):
        communities[community].add(node)

    return list(communities.values())


def get_all_edges(nodes):
    return chain(combinations(nodes, 2), ((u, u) for u in nodes))


########
# PHASES
########


def run_first_phase(node_to_comm, adj_matrix, n, force_merge=False):
    M = modularity_matrix(adj_matrix)
    best_node_to_comm = node_to_comm.copy()
    num_communities = len(set(best_node_to_comm))
    is_updated = not (n and num_communities == n)

    # QUESTION: Randomize the order of the nodes before iterating?

    while is_updated:
        is_updated = False
        for i, neighbors in enumerate(adj_matrix):
            num_communities = len(set(best_node_to_comm))
            if n and num_communities == n:
                break

            best_Q = modularity(M, invert_node_to_comm(best_node_to_comm))
            max_delta_Q = 0.0
            updated_node_to_comm, visited_communities = best_node_to_comm, set()
            for j, weight in enumerate(neighbors):
                # Skip if self-loop or not neighbor
                if i == j or not weight:
                    continue

                neighbor_comm = best_node_to_comm[j]
                if neighbor_comm in visited_communities:
                    continue

                # Remove node i from its community and place it in the community
                # of its neighbor j
                candidate_node_to_comm = best_node_to_comm.copy()
                candidate_node_to_comm[i] = neighbor_comm

                candidate_Q = modularity(
                    M,
                    invert_node_to_comm(candidate_node_to_comm)
                )
                delta_Q = candidate_Q - best_Q
                if delta_Q > max_delta_Q or (force_merge and not max_delta_Q):
                    updated_node_to_comm = candidate_node_to_comm
                    max_delta_Q = delta_Q

                visited_communities.add(neighbor_comm)

            if best_node_to_comm != updated_node_to_comm:
                best_node_to_comm = updated_node_to_comm
                is_updated = True

    return best_node_to_comm


def run_second_phase(node_to_comm, adj_matrix, true_partition):
    comm_to_nodes = defaultdict(lambda: [])
    for i, comm in enumerate(node_to_comm):
        comm_to_nodes[comm].append(i)
    node_clusters = list(comm_to_nodes.values())

    new_adj_matrix, new_true_partition = [], []
    for i, cluster in enumerate(node_clusters):
        true_cluster = {v for u in cluster for v in true_partition[u]}
        row_vec = []
        for j, neighbor_cluster in enumerate(node_clusters):
            if i == j:  # Sum all intra-community weights and add as self-loop
                edge_weights = (adj_matrix[u][v]
                                for u, v in get_all_edges(cluster))
                edge_weight = 2 * sum(edge_weights)
            else:
                edge_weights = (adj_matrix[u][v]
                                for u in cluster for v in neighbor_cluster)
                edge_weight = sum(edge_weights)

            row_vec.append(edge_weight)

        new_true_partition.append(true_cluster)
        new_adj_matrix.append(row_vec)

    # TODO: Use numpy more efficiently
    return np.array(new_adj_matrix), new_true_partition


######
# MAIN
######


def louvain_method(adj_matrix, n=None):
    optimal_adj_matrix = adj_matrix
    node_to_comm = initialize_node_to_comm(adj_matrix)
    true_partition = [{i} for i in range(len(adj_matrix))]

    is_optimal = False
    while not is_optimal:
        optimal_node_to_comm = run_first_phase(
            node_to_comm,
            optimal_adj_matrix,
            n
        )

        if optimal_node_to_comm == node_to_comm:
            if not n:
                break

            optimal_node_to_comm = run_first_phase(
                node_to_comm,
                optimal_adj_matrix,
                n,
                force_merge=True
            )

        optimal_adj_matrix, true_partition = run_second_phase(
            optimal_node_to_comm,
            optimal_adj_matrix,
            true_partition
        )

        if n and len(true_partition) == n:
            break

        node_to_comm = initialize_node_to_comm(optimal_adj_matrix)

    return true_partition
