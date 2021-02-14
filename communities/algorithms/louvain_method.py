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
    ani_frames = [{"C": best_node_to_comm, "Q": 0.0}]

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

                    ani_frames.append({
                        "C": candidate_node_to_comm,
                        "Q": candidate_Q
                    })

                visited_communities.add(neighbor_comm)

            # Set Q for first frame
            if not i and ani_frames[0]["C"] == best_node_to_comm:
                ani_frames[0]["Q"] = best_Q

            if best_node_to_comm != updated_node_to_comm:
                best_node_to_comm = updated_node_to_comm
                is_updated = True

    if ani_frames[-1]["C"] != best_node_to_comm:
        ani_frames.append({"C": best_node_to_comm, "Q": best_Q})

    return best_node_to_comm, ani_frames


def run_second_phase(node_to_comm, adj_matrix, true_partition, true_comms):
    comm_to_nodes = defaultdict(lambda: [])
    for i, comm in enumerate(node_to_comm):
        comm_to_nodes[comm].append(i)
    comm_to_nodes = list(comm_to_nodes.items())

    new_adj_matrix, new_true_partition = [], []
    for i, (comm, nodes) in enumerate(comm_to_nodes):
        true_nodes = {v for u in nodes for v in true_partition[u]}
        true_comms[i] = true_comms[comm]

        row_vec = []
        for j, (_, neighbors) in enumerate(comm_to_nodes):
            if i == j:  # Sum all intra-community weights and add as self-loop
                edge_weights = (adj_matrix[u][v]
                                for u, v in get_all_edges(nodes))
                edge_weight = 2 * sum(edge_weights)
            else:
                edge_weights = (adj_matrix[u][v]
                                for u in nodes for v in neighbors)
                edge_weight = sum(edge_weights)

            row_vec.append(edge_weight)

        new_true_partition.append(true_nodes)
        new_adj_matrix.append(row_vec)

    # TODO: Use numpy more efficiently
    return np.array(new_adj_matrix), new_true_partition, true_comms


######
# MAIN
######


def louvain_method(adj_matrix : np.ndarray, n : int = None) -> list:
    optimal_adj_matrix = adj_matrix
    node_to_comm = initialize_node_to_comm(adj_matrix)
    true_partition = [{i} for i in range(len(adj_matrix))]
    true_comms = {c: c for c in node_to_comm}

    M = modularity_matrix(adj_matrix)
    def update_frame(frame, partition, comm_aliases, recalculate_Q):
        true_node_to_comm = list(range(len(adj_matrix)))
        for i, community in enumerate(frame["C"]):
            for node in partition[i]:
                true_node_to_comm[node] = comm_aliases[community]

        frame["C"] = true_node_to_comm
        if recalculate_Q:
            frame["Q"] = modularity(M, invert_node_to_comm(frame["C"]))

        return frame

    ani_frames = []
    is_optimal = False
    while not is_optimal:
        optimal_node_to_comm, frames = run_first_phase(
            node_to_comm,
            optimal_adj_matrix,
            n
        )

        if optimal_node_to_comm == node_to_comm:
            if not n:
                frames = (update_frame(f, true_partition, true_comms, bool(ani_frames)) for f in frames)
                ani_frames.extend(frames)
                break

            optimal_node_to_comm, frames = run_first_phase(
                node_to_comm,
                optimal_adj_matrix,
                n,
                force_merge=True
            )

        frames = (update_frame(f, true_partition, true_comms, bool(ani_frames)) for f in frames)
        ani_frames.extend(frames)

        optimal_adj_matrix, true_partition, true_comms = run_second_phase(
            optimal_node_to_comm,
            optimal_adj_matrix,
            true_partition,
            true_comms
        )

        if n and len(true_partition) == n:
            break

        node_to_comm = initialize_node_to_comm(optimal_adj_matrix)

    return true_partition, ani_frames
