# Standard Library
import random
from collections import defaultdict

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    # TODO: Make scales dynamic
    pos_communities = _position_communities(g, partition, scale=8., seed=2)
    pos_nodes = _position_nodes(g, partition, scale=1., seed=2)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition)
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in enumerate(partition):
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):
    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = defaultdict(list)
    for node, community in enumerate(partition):
        communities[community].append(node)

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


class AlgoAnimation(object):
    def __init__(self, adj_matrix, frames, seed=0):
        np.random.seed(seed)
        random.seed(seed)

        self.fig = plt.figure()

        self.G = nx.from_numpy_matrix(adj_matrix)

        self.source_pos = nx.spring_layout(self.G)
        self.target_pos = community_layout(self.G, frames[-1]["C"])

        minima, maxima = 0, len(adj_matrix) - 1
        norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        self.color_mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        # self.temp_frames = frames # TEMP: For testing purposes
        self.frames = self._interpolate_transition_frames(frames)
        self.frame_indices = list(self._generate_frame_indices())

    def _interpolate_transition_frames(self, frames):
        mean_eucl_dists = []
        for i in range(len(frames) - 1):
            source_pos = community_layout(self.G, frames[i]["C"])
            targ_pos = community_layout(self.G, frames[i + 1]["C"])

            eucl_dists = []
            for node in range(self.G.number_of_nodes()):
                eucl_dist = np.linalg.norm(source_pos[node]-targ_pos[node])
                eucl_dists.append(eucl_dist)

            mean_eucl_dists.append(max(eucl_dists))

        D = np.array(mean_eucl_dists)
        min_n_frames, max_n_frames = 2, 11 # TODO: These numbers are arbitrary; make dynamic
        norm_dists = (D - D.min()) / D.ptp() * (max_n_frames - min_n_frames) + min_n_frames

        frames_with_transitions = []
        for i, num_trans_frames in zip(range(len(frames) - 1), norm_dists):
            source_pos = community_layout(self.G, frames[i]["C"])
            targ_pos = community_layout(self.G, frames[i + 1]["C"])

            num_trans_frames = int(num_trans_frames) - 1
            for j in range(num_trans_frames + 1):
                frame = frames[i].copy()
                frame["pos"] = {}
                for n in range(self.G.number_of_nodes()):
                    frame["pos"][n] = source_pos[n] * (1 - j / (num_trans_frames)) + targ_pos[n] * j / (num_trans_frames)

                if j > 0: # QUESTION: Is this really needed?
                    frame["C"] = frames[i + 1]["C"]

                frames_with_transitions.append(frame)

        last_frame = frames[-1].copy()
        last_frame["pos"] = community_layout(self.G, frames[-1]["C"])

        return frames_with_transitions + [last_frame]

    def _generate_frame_indices(self):
        num_frames = len(self.frames)
        for i in range(num_frames):
            yield i

        for _ in range(int(0.33 * num_frames)):
            yield i

    def _update(self, i):
        plt.clf()
        plt.gca().set_facecolor("black")

        C = self.frames[i]["C"]
        Q = self.frames[i]["Q"]
        pos = self.frames[i]["pos"]

        # TODO: Add "input graph" frame to beginning
        # TODO: Assign initial communities 1-by-1
        # TODO: Add patches and update title on last frame
        # TODO: Make node size dynamic
        # TODO: Add Q graph and titles
        # TODO: Make edge alpha depend on weight

        nodes = nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            node_color=[self.color_mapper.to_rgba(c) for c in C],
            linewidths=1.0
        )
        nodes.set_edgecolor("w")
        nx.draw_networkx_edges(
            self.G,
            pos=pos,
            # edge_color=(0.85, 0.85, 0.85)
            edge_color=(1.0, 1.0, 1.0, 0.75)
        )

    def show(self, duration=15, filename=None, dpi=None):
        ani = FuncAnimation(
            self.fig,
            self._update,
            frames=self.frame_indices,
            init_func=lambda: self._update(0),
            interval=int(15000 / len(self.frame_indices)),
            repeat=True,
            save_count=len(self.frame_indices)
        )

        if not filename:
            plt.show()
        else:
            savefig_kwargs={'facecolor':'black'}
            ani.save(
                filename,
                fps=int(len(self.frame_indices) / 15),
                dpi=dpi,
                savefig_kwargs={"facecolor": "black"}
            )

    def plot_single_frame(self, i): # TEMP: For testing purposes
        plt.gca().set_facecolor("black")

        C = self.temp_frames[i]["C"]
        pos = community_layout(self.G, C)

        print(C)

        nodes = nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            node_color=[self.color_mapper.to_rgba(c) for c in C],
            linewidths=1.0
        )
        nodes.set_edgecolor("w")
        nx.draw_networkx_edges(
            self.G,
            pos=pos,
            # edge_color=(0.85, 0.85, 0.85)
            edge_color=(1.0, 1.0, 1.0, 0.75)
        )
        plt.show()
