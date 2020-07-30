# Standard Library
from itertools import combinations, chain
from collections import defaultdict


#########
# HELPERS
#########


def get_all_edges(nodes):
    return chain(combinations(nodes, 2), ((u, u) for u in nodes))


def compute_sigma_in(nodes_in_C, adj_matrix):
    # Sum of the edge weights of the links inside community C
    return sum(adj_matrix[u][v] for u, v in get_all_edges(nodes_in_C))


def compute_sigma_tot(nodes_in_C, adj_matrix):
    # Sum of all the weights of the links *to* nodes in a community
    sigma_tot = 0.0
    for node in nodes_in_C:
        for neighbor, edge_weight in enumerate(adj_matrix[node]):
            if not neighbor in nodes_in_C:
                sigma_tot += edge_weight

    return sigma_tot


def compute_k(node, adj_matrix):
    # Weighted degree of node
    return sum(adj_matrix[node])


def compute_k_in(node, nodes_in_C, adj_matrix):
    # Sum of the weights of the links between a node and all other nodes in a
    # community
    k_i_in = 0.0
    for neighbor in nodes_in_C:
        k_i_in += adj_matrix[node][neighbor]

    return k_i_in


def create_Q_computer(adj_matrix):
    m = compute_sigma_in(range(len(adj_matrix)), adj_matrix)

    summation_terms = []
    for i, j in get_all_edges(range(len(adj_matrix))):
        A_ij = adj_matrix[i][j]
        k_i = compute_k(i, adj_matrix)
        k_j = compute_k(j, adj_matrix)

        term = A_ij - ((k_i * k_j) / (2 * m))
        summation_terms.append((i, j, term))

    def compute_Q(node_to_comm):
        Q = 0.0
        for i, j, term in summation_terms:
            delta_ij = int(node_to_comm[i] == node_to_comm[j])
            Q += term * delta_ij

        return Q * (1 / (2 * m))

    return compute_Q


def initialize_communities(adj_matrix):
    return list(range(len(adj_matrix)))


########
# PHASES
########


def run_first_phase(node_to_comm, adj_matrix, size, force_merge=False):
    """
    For each node i, the change in modularity is computed for removing i from
    its own community and moving it into the community of each neighbor j of i.
    Once this value, delta_Q, is calculated for all communities i is connected
    to, i is placed in the community that resulted in the largest delta_Q. If
    no increase is possible, i remains in its original community. This process
    is applied repeatedly and sequentially to all nodes until no modularity
    increase can occur. Once this local maximum of modularity is hit, the first
    phase has ended. (From: https://en.wikipedia.org/wiki/Louvain_modularity#Algorithm)
    """

    compute_Q = create_Q_computer(adj_matrix)
    best_node_to_comm = node_to_comm.copy()
    num_communities = len(set(best_node_to_comm))
    is_updated = not (size and num_communities == size)

    # QUESTION: Randomize the order of the nodes before iterating?

    while is_updated:
        is_updated = False
        for i, neighbors in enumerate(adj_matrix):
            num_communities = len(set(best_node_to_comm))
            if size and num_communities == size:
                break

            best_Q, max_delta_Q = compute_Q(best_node_to_comm), 0.0
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

                candidate_Q = compute_Q(candidate_node_to_comm)
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
    """
    All nodes in the same community are grouped together and a new network is
    built where nodes are the communities from the previous phase. Any links
    between nodes of the same community are now represented by self-loops on
    the new community node and links from multiple nodes in the same community
    to a node in a different community are represented by weighted edges
    between communities. (From: https://en.wikipedia.org/wiki/Louvain_modularity#Algorithm)
    """

    comm_to_nodes = defaultdict(lambda: [])
    for i, comm in enumerate(node_to_comm):
        comm_to_nodes[comm].append(i)
    node_clusters = list(comm_to_nodes.values())

    new_adj_matrix, new_true_partition = [], []
    for i, cluster in enumerate(node_clusters):
        true_cluster = [v for u in cluster for v in true_partition[u]]
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

    return new_adj_matrix, new_true_partition


######
# MAIN
######


def louvain_modularity(adj_matrix, size=None):
    optimal_adj_matrix = adj_matrix
    node_to_comm = initialize_communities(adj_matrix)
    true_partition = [[i] for i in range(len(adj_matrix))]

    is_optimal = False
    while not is_optimal:
        optimal_node_to_comm = run_first_phase(
            node_to_comm,
            optimal_adj_matrix,
            size
        )

        if optimal_node_to_comm == node_to_comm:
            if not size:
                break

            optimal_node_to_comm = run_first_phase(
                node_to_comm,
                optimal_adj_matrix,
                size,
                force_merge=True
            )

        optimal_adj_matrix, true_partition = run_second_phase(
            optimal_node_to_comm,
            optimal_adj_matrix,
            true_partition
        )

        if size and len(true_partition) == size:
            break

        node_to_comm = initialize_communities(optimal_adj_matrix)

    return true_partition
