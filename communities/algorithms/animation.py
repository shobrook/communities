# Standard Library
import random
from collections import defaultdict, namedtuple
from functools import partial

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

Artists = namedtuple(
    "Artists",
    (
        "network_nodes",
        "network_edges",
        "modularity_line"
        # "title_1",
        # "title_2"
    )
)


################
# CLUSTER LAYOUT
################


def _inter_cluster_edges(G, partition):
    edges = defaultdict(list)

    for (i, j) in G.edges():
        c_i = partition[i]
        c_j = partition[j]

        if c_i == c_j:
            continue

        edges[(c_i, c_j)].append((i, j))

    return edges


def _position_clusters(G, partition, **kwargs):
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(set(partition))

    inter_cluster_edges = _inter_cluster_edges(G, partition)
    for (c_i, c_j), edges in inter_cluster_edges.items():
        hypergraph.add_edge(c_i, c_j, weight=len(edges)) # TODO: Try setting weight=1

    pos_clusters = nx.circular_layout(hypergraph, scale=4.0)
    # pos_clusters = nx.spring_layout(hypergraph, **kwargs)

    return pos_clusters


def _position_nodes(G, partition, **kwargs):
    clusters = defaultdict(list)
    for node, community in enumerate(partition):
        clusters[community].append(node)

    pos = {}
    for c_i, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        if "pos" in kwargs:
            init_pos = {n: kwargs["pos"][n] for n in nodes if n in kwargs["pos"]}
            init_fixed = list(init_pos.keys())
            pos_subgraph = nx.spring_layout(
                subgraph,
                **{
                    **kwargs,
                    **{
                        "pos": init_pos if init_pos else None,
                        "fixed": init_fixed if "fixed" not in kwargs and init_fixed else None
                    }
                }
            )
        else:
            pos_subgraph = nx.spring_layout(subgraph, **kwargs)

        pos.update(pos_subgraph)

    return pos


def cluster_layout(G, pos_nodes, pos_clusters):
    pos = {}
    for node in G.nodes():
        pos[node] = pos_clusters[node] + pos_nodes[node]

    return pos


###############
# INTERPOLATION
###############


def _pos_endpoints(G, frames, seed):
    prev_pos_clusters = None
    prev_pos_nodes = None

    pos_endpoints = []
    for i in range(len(frames) - 1):
        partition = frames[i]["C"] # TODO: Change 'C' to 'partition'
        next_partition = frames[i + 1]["C"]

        if not prev_pos_clusters and not prev_pos_nodes:
            prev_pos_clusters = _position_clusters(G, partition, scale=4.0, seed=seed)
            prev_pos_nodes = _position_nodes(G, partition, scale=0.5, seed=seed)

        source_pos = cluster_layout(
            G,
            prev_pos_nodes,
            {i: prev_pos_clusters[c_i] for i, c_i in enumerate(partition)}
        )

        init_pos_nodes = {}
        for node in G.nodes():
            if partition[node] != next_partition[node]:
                continue

            init_pos_nodes[node] = prev_pos_nodes[node]

        mid_pos_nodes = _position_nodes(G, partition, pos=init_pos_nodes, scale=0.5, seed=seed)
        mid_pos = cluster_layout(
            G,
            mid_pos_nodes,
            {i: prev_pos_clusters[c_i] for i, c_i in enumerate(partition)}
        )

        target_pos_nodes = _position_nodes(G, partition, pos=mid_pos_nodes, fixed=None, scale=0.5, seed=seed)
        target_pos_clusters = _position_clusters(G, partition, pos={n: coord for n, coord in prev_pos_clusters.items() if n in next_partition}, scale=4.0, seed=seed)
        target_pos = cluster_layout(
            G,
            target_pos_nodes,
            {i: target_pos_clusters[c_i] for i, c_i in enumerate(next_partition)}
        )

        prev_pos_clusters = target_pos_clusters
        prev_pos_nodes = target_pos_nodes

        pos_endpoints.append((source_pos, mid_pos, target_pos))

    # init_endpoint = (
    #     nx.spring_layout(G, scale=4.0, seed=seed),
    #     nx.spring_layout(G, scale=4.0, seed=seed),
    #     pos_endpoints[0][0]
    # )
    # pos_endpoints.insert(0, init_endpoint)

    return pos_endpoints


def _max_euclidean_distance(source_pos, target_pos):
    distances = []
    for node in source_pos.keys():
        distance = np.linalg.norm(target_pos[node] - source_pos[node])
        distances.append(distance)

    return max(distances)


def _transition_lengths(frames, pos_endpoints):
    trans_distances = []
    for i in range(len(frames) - 1):
        source_pos, mid_pos, target_pos = pos_endpoints[i]
        mid_distance = _max_euclidean_distance(source_pos, mid_pos)
        target_distance = _max_euclidean_distance(mid_pos, target_pos)

        trans_distances.extend([mid_distance, target_distance])

    D = np.array(trans_distances)
    n_min, n_max = 2, 11 # TODO: These numbers are arbitrary; make dynamic
    norm_D = (D - D.min()) / D.ptp() * (n_max - n_min) + n_min

    trans_lengths = ((norm_D[i], norm_D[i + 1]) for i in range(0, len(norm_D), 2))
    return [(int(mid_l) - 1, int(targ_l) - 1) for mid_l, targ_l in trans_lengths]


def _interpolate_frames(G, base_frame, source_pos, target_pos, interpol_len):
    for j in range(interpol_len + 1):
        frame = base_frame.copy()
        frame["pos"] = {}
        for n in range(G.number_of_nodes()):
            frame["pos"][n] = source_pos[n] * (1 - j / interpol_len)
            frame["pos"][n] += target_pos[n] * j / interpol_len

        yield frame


def interpolate(G, frames, seed):
    pos_endpoints = _pos_endpoints(G, frames, seed)
    trans_lengths = _transition_lengths(frames, pos_endpoints)

    interpolated_frames = []
    _iter_batch = zip(range(len(frames) - 1), pos_endpoints, trans_lengths) # len(frames) - 1
    for i, (source_pos, mid_pos, target_pos), (mid_len, targ_len) in _iter_batch:
        mid_frames = _interpolate_frames(
            G,
            {**frames[i], **{"index": i}},
            source_pos,
            mid_pos,
            mid_len
        )
        target_frames = _interpolate_frames(
            G,
            {**frames[i], **{"C": frames[i + 1]["C"], "index": i}},
            mid_pos,
            target_pos,
            targ_len
        )

        interpolated_frames.extend(mid_frames)
        interpolated_frames.extend(target_frames)

    last_frame = frames[-1].copy()
    last_frame["pos"] = target_pos
    last_frame["index"] = len(frames) - 1

    return interpolated_frames + [last_frame]


###########
# ANIMATION
###########


class AlgoAnimation(object):
    def __init__(self, A, frames, seed=0):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.G = nx.from_numpy_matrix(A)
        self.interpolated_frames = interpolate(self.G, frames, self.seed)

        self.x = list(range(len(frames)))
        self.y = [frame["Q"] for frame in frames]

        plt.rcParams["figure.facecolor"] = "black"
        plt.rcParams["axes.facecolor"] = "black"

        self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2, figsize=(12, 6))
        self.artists = None

    def _calculate_axes_limits(self, node_size, node_border_size):
        x, y = [], []
        for frame in self.interpolated_frames:
            for coordinate in frame["pos"].values():
                x.append(coordinate[0])
                y.append(coordinate[1])

        xlim = [min(x), max(x)]
        ylim = [min(y), max(y)]

        xy_pixels = self.ax0.transData.transform(np.vstack([xlim, ylim]).T)
        xpix, ypix = xy_pixels.T

        offset = node_size + (node_border_size * 2) + 100
        xlim_pix = [xpix[0] - offset, xpix[1] + offset]
        ylim_pix = [ypix[0] - offset, ypix[1] + offset]

        xy_coords = self.ax0.transAxes.inverted().transform(np.vstack([xlim_pix, ylim_pix]).T)
        xlim, ylim = xy_coords.T

        return xlim, ylim

    def init_fig(self):
        self.ax0.set_title("Input Network", color="white"),
        self.ax1.set_title("Modularity (Q)", color="white")

        self.ax1.set_xlim([0.0, self.x[-1]])
        self.ax1.set_ylim([0.0, max(self.y)])

        for spine in self.ax1.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.75))

        self.ax1.xaxis.label.set_color((1.0, 1.0, 1.0, 0.75))
        self.ax1.yaxis.label.set_color((1.0, 1.0, 1.0, 0.75))
        self.ax1.tick_params(axis="y", colors=(1.0, 1.0, 1.0, 0.75))
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        plt.tight_layout(pad=3.0)

        xlim, ylim = self._calculate_axes_limits(200, 1)
        self.ax0.set_xlim(xlim)
        self.ax0.set_ylim(ylim)

        self.artists = Artists(
            nx.draw_networkx_nodes(
                self.G,
                pos=self.interpolated_frames[0]["pos"],
                node_color=self.interpolated_frames[0]["C"], # QUESTION: Should they all be the same color initially?
                linewidths=1.0,
                ax=self.ax0,
                cmap=cm.jet # TODO: Fix so some colors aren't so similar
            ),
            nx.draw_networkx_edges(
                self.G,
                pos=self.interpolated_frames[0]["pos"],
                edge_color=(1.0, 1.0, 1.0, 0.75),
                ax=self.ax0
            ),
            self.ax1.plot([], [], color="white")[0]
            # self.ax0.set_title("Input Graph", color="white"),
            # self.ax1.set_title("Modularity (Q) = 0.0", color="white")
        )
        self.artists.network_nodes.set_edgecolor("w")

        return self.artists

    def frame_iter(self):
        num_frames = len(self.interpolated_frames)

        # First frame (input graph) should be displayed for 12.5% of the
        # animation
        # for _ in range(int(0.05 * num_frames)):
        #     yield 0

        for i in range(1, num_frames - 1):
            yield i

        # for _ in range(int(0.15 * num_frames)):
        #     yield num_frames - 1

    def update(self, i):
        Q = self.interpolated_frames[i]["Q"]
        partition = self.interpolated_frames[i]["C"]
        pos = self.interpolated_frames[i]["pos"]
        index = self.interpolated_frames[i]["index"]

        # self.artists.title_1.set_text(f"Iteration #{index}")
        # self.artists.title_2.set_text(f"Modularity (Q) = {Q}")

        offsets = [0.0 for _ in range(len(pos.keys()))]
        for node, coord in pos.items():
            offsets[node] = coord

        self.artists.network_nodes.set_offsets(offsets)
        self.artists.network_nodes.set_array(np.array(partition))

        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in self.G.edges()])
        self.artists.network_edges.set_verts(edge_pos)

        self.artists.modularity_line.set_data(self.x[:index], self.y[:index])

        return self.artists

    def show(self, duration=15, filename=None, dpi=None):
        num_frames = len(list(self.frame_iter()))
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frame_iter,
            init_func=self.init_fig,
            # interval=1000,
            save_count=num_frames,
            blit=True
        )

        if not filename:
            plt.show()
        else:
            anim.save(
                filename,
                fps=int(num_frames / duration),
                dpi=dpi
            )
