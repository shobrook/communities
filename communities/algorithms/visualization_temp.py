# Standard Library
import random
from collections import defaultdict

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation


#########
# HELPERS
#########


def _position_communities(G, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(G, partition)

    communities = set(partition)
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    if kwargs["pos"]:
        kwargs["pos"] = {n: kwargs["pos"][n] for n in communities if n in kwargs["pos"]}
        pos_communities = nx.spring_layout(
            hypergraph,
            fixed=list(kwargs["pos"].keys()),
            **kwargs
        )
    else:
        pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in enumerate(partition):
        pos[node] = pos_communities[community]

    return pos, pos_communities


def _find_between_community_edges(G, partition):
    edges = dict()

    for (ni, nj) in G.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(G, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = defaultdict(list)
    for node, community in enumerate(partition):
        communities[community].append(node)

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        if kwargs["pos"]:
            init_pos = {n: kwargs["pos"][n] for n in nodes if n in kwargs["pos"]}
            pos_subgraph = nx.spring_layout(
                subgraph,
                fixed=list(init_pos.keys()),
                **{**kwargs, **{"pos": init_pos}}
            )
        else:
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)

        pos.update(pos_subgraph)

    return pos


###########
# UTILITIES
###########


def community_layout(G, partition, community_pos=None, node_pos=None, seed=0):
    # TODO: Make scales dynamic
    pos_communities, community_pos = _position_communities(
        G,
        partition,
        scale=8.0,
        seed=seed,
        pos=community_pos
    )
    pos_nodes = _position_nodes(
        G,
        partition,
        scale=1.0,
        seed=seed,
        pos=node_pos
    )

    pos = dict()
    for node in G.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos, community_pos, pos_nodes


def create_color_mapper(num_nodes):
    minima, maxima = 0, num_nodes - 1
    norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    return cm.ScalarMappable(norm=norm, cmap=cm.jet)

    # TODO: Make this better.. some of the colors are too similar to each other


##########
# ANIMATOR
##########


class AlgoAnimation(object):
    def __init__(self, A, frames, seed=0):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.G = nx.from_numpy_matrix(A)
        self.frames = self._interpolate_frames(frames)
        self.frame_indices = list(self._generate_frame_indices())

        plt.rcParams["figure.facecolor"] = "black"
        plt.rcParams["axes.facecolor"] = "black"

        self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2, figsize=(12, 6))

        self.x = list(range(len(frames)))
        self.y = [frame["Q"] for frame in frames]

        self.ax1.set_xlim([self.x[0], self.x[-1]])
        self.ax1.set_ylim([0.0, max(self.y)])

        for spine in self.ax1.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.75))

        self.ax1.xaxis.label.set_color((1.0, 1.0, 1.0, 0.75))
        self.ax1.yaxis.label.set_color((1.0, 1.0, 1.0, 0.75))
        self.ax1.tick_params(axis="y", colors=(1.0, 1.0, 1.0, 0.75))
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        plt.tight_layout(3.0)

        self.color_mapper = create_color_mapper(len(A))

    ## Frame Preprocessing ##

    def _generate_frame_indices(self):
        num_frames = len(self.frames)

        # First frame (input graph) should be displayed for 12.5% of the
        # animation
        for _ in range(int(0.165 * num_frames)):
            yield 0

        for i in range(1, num_frames):
            yield i

        # Last frame (partitioned graph) should be displayed for 25% of the
        # animation
        for _ in range(int(0.33 * num_frames)):
            yield i

    def _compute_trans_sequence(self, frame, next_frame):
        C = frame["C"]
        C_next = next_frame["C"]

        src_pos, community_pos, node_pos = community_layout(
            self.G,
            C,
            seed=self.seed
        )
        for i in range(len(C)):
            if C[i] == C_next[i]:
                continue

            del node_pos[i]
        mid_pos, _, _ = community_layout(
            self.G,
            C_next,
            community_pos,
            node_pos,
            seed=self.seed
        )
        targ_pos, _, _ = community_layout(self.G, C_next, seed=self.seed)

        # TODO: Check that this is actually working

        return src_pos, mid_pos, targ_pos

    def _compute_max_euclidean_dist(self, src_pos, targ_pos):
        eucl_dists = []
        for node in range(self.G.number_of_nodes()):
            eucl_dist = np.linalg.norm(targ_pos[node] - src_pos[node])
            eucl_dists.append(eucl_dist)

        return max(eucl_dists)

    def _compute_num_trans_frames(self, frames):
        max_trans_dists = []
        for i in range(len(frames) - 1):
            src_pos, mid_pos, targ_pos = self._compute_trans_sequence(frames[i], frames[i + 1])

            mid_dist = self._compute_max_euclidean_dist(src_pos, mid_pos)
            targ_dist = self._compute_max_euclidean_dist(mid_pos, targ_pos)

            max_trans_dists.extend([mid_dist, targ_dist])

        D = np.array(max_trans_dists)
        n_min, n_max = 2, 11 # TODO: These numbers are arbitrary; make dynamic
        norm_D = (D - D.min()) / D.ptp() * (n_max - n_min) + n_min

        return [(norm_D[i], norm_D[i + 1]) for i in range(0, len(norm_D), 2)]

    def _create_transition_frames(self, base_frame, src_pos, targ_pos, num_frames, next_C=None):
        for j in range(num_frames + 1):
            frame = base_frame.copy()
            frame["pos"] = {}
            for n in range(self.G.number_of_nodes()):
                frame["pos"][n] = src_pos[n] * (1 - j / num_frames) + targ_pos[n] * j / num_frames

            if j > 0 and next_C:
                frame["C"] = next_C

            yield frame

    def _interpolate_frames(self, frames):
        T = self._compute_num_trans_frames(frames)

        frames_with_transitions = []
        for i, T_ij in zip(range(len(frames) - 1), T):
            T_i, T_j = T_ij
            src_pos, mid_pos, targ_pos = self._compute_trans_sequence(frames[i], frames[i + 1])

            mid_frames = self._create_transition_frames(
                {**frames[i], **{"index": i}},
                src_pos,
                mid_pos,
                int(T_i) - 1,
                frames[i + 1]["C"]
            )
            targ_frames = self._create_transition_frames(
                {**frames[i], **{"C": frames[i + 1]["C"], "index": i}},
                mid_pos,
                targ_pos,
                int(T_j) - 1
            )

            frames_with_transitions.extend(mid_frames)
            frames_with_transitions.extend(targ_frames)

        last_frame = frames[-1].copy()
        last_frame["pos"] = targ_pos
        last_frame["index"] = len(frames) - 1

        return frames_with_transitions + [last_frame]

    ## Animation ##

    def _update(self, i):
        self.ax0.clear()
        self.ax1.clear()

        C = self.frames[i]["C"]
        Q = self.frames[i]["Q"]
        pos = self.frames[i]["pos"]
        index = self.frames[i]["index"]

        if not i:
            self.ax0.set_title("Input Graph", color="white")
        else:
            self.ax0.set_title(f"Iteration #{index}", color="white")

        self.ax1.set_title(f"Modularity (Q) = {Q}", color="white")
        self.ax1.plot(self.x[:index], self.y[:index], color="white")

        # TODO: Add patches and update title on last frame
            # The patches bounding the communities can be made by finding the
            # positions of the nodes for each community and then drawing a patch
            # (e.g. matplotlib.patches.Circle) that contains all positions (and
            # then some).
        # TODO: Make node size dynamic
        # TODO: Fix Q values
        # TODO: Make edge alpha depend on weight
        # TODO: Blitting!

        nodes = nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            node_color=[self.color_mapper.to_rgba(c) for c in C],
            linewidths=1.0,
            ax=self.ax0
        )
        nodes.set_edgecolor("w")
        edges = nx.draw_networkx_edges(
            self.G,
            pos=pos,
            edge_color=(1.0, 1.0, 1.0, 0.75),
            ax=self.ax0
        )

    def show(self, duration=15, filename=None, dpi=None):
        print(len(self.frame_indices))
        ani = FuncAnimation(
            self.fig,
            self._update,
            frames=self.frame_indices,
            init_func=lambda: self._update(0),
            interval=int((duration * 1000) / len(self.frame_indices)),
            repeat=True,
            save_count=len(self.frame_indices)
        )

        if not filename:
            plt.show()
        else:
            ani.save(
                filename,
                fps=int(len(self.frame_indices) / duration),
                dpi=dpi
            )
