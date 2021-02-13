# Standard Library
import random
from collections import defaultdict, namedtuple

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

Artists = namedtuple("Artists", ("network_nodes", "network_edges", "modularity_line", "ax0_title", "ax1_title"))
HYPERGRAPH_SCALE = 16.0 # 4.0
SUBGRAPH_SCALE = 4.0 # 0.5
GREY = (1.0, 1.0, 1.0, 0.75)
TITLE_GREY = (0.15, 0.15, 0.15, 1.0)
DARK_GREY = (0.3, 0.3, 0.3, 1.0)
LIGHT_GREY = (0.6, 0.6, 0.6, 1.0)


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
        hypergraph.add_edge(c_i, c_j, weight=len(edges))

    pos_clusters = nx.circular_layout(hypergraph, scale=HYPERGRAPH_SCALE)
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
                        "fixed": init_fixed if "fixed" not in kwargs and init_fixed else None,
                        "scale": SUBGRAPH_SCALE
                    }
                }
            )
        else:
            pos_subgraph = nx.spring_layout(subgraph, **{**kwargs, **{"scale": SUBGRAPH_SCALE}})

        pos.update(pos_subgraph)

    return pos


def cluster_layout(G, pos_nodes, pos_clusters):
    pos = {}
    # print(pos_nodes)
    # print(pos_clusters)
    # print()
    for node in G.nodes():
        pos[node] = pos_nodes[node] + pos_clusters[node]

    return pos


###############
# INTERPOLATION
###############


def _pos_endpoints(G, frames, seed):
    prev_pos_clusters = None
    prev_pos_nodes = None

    pos_endpoints = []
    for i in range(len(frames) - 1):
        partition = frames[i]["C"]
        next_partition = frames[i + 1]["C"]

        if not prev_pos_clusters and not prev_pos_nodes:
            prev_pos_clusters = _position_clusters(G, partition)
            prev_pos_nodes = _position_nodes(G, partition, seed=seed)

        _prev_pos_clusters = {i: prev_pos_clusters[c_i] for i, c_i in enumerate(partition)}
        source_pos = cluster_layout(G, prev_pos_nodes, _prev_pos_clusters)

        init_pos_nodes = {}
        for node in G.nodes():
            if partition[node] != next_partition[node]:
                continue

            init_pos_nodes[node] = prev_pos_nodes[node]

        mid_pos_nodes = _position_nodes(G, partition, pos=init_pos_nodes, seed=seed) # partition
        mid_pos = cluster_layout(G, mid_pos_nodes, _prev_pos_clusters)

        target_pos_nodes = _position_nodes(G, partition, pos=mid_pos_nodes, fixed=None, seed=seed) # partition
        target_pos_clusters = _position_clusters(G, partition) # partition
        target_pos = cluster_layout(
            G,
            target_pos_nodes,
            {i: target_pos_clusters[c_i] for i, c_i in enumerate(next_partition)}
        )

        prev_pos_clusters = target_pos_clusters
        prev_pos_nodes = target_pos_nodes

        pos_endpoints.append((source_pos, mid_pos, target_pos))

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
    n_min, n_max = 2, int(len(frames) / 7)
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
    _iter_batch = zip(range(len(frames) - 1), pos_endpoints, trans_lengths)
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

    return interpolated_frames + [last_frame, last_frame]


###########
# ANIMATION
###########


class Animation(object):
    def __init__(self, A, frames, seed=2, dark=True):
        np.random.seed(seed)
        random.seed(seed)

        self.G = nx.from_numpy_matrix(A)
        self.interpolated_frames = interpolate(self.G, frames, seed)

        self.x = list(range(len(frames)))
        self.y = [frame["Q"] for frame in frames]

        self.is_dark = dark
        plt.rcParams["figure.facecolor"] = "black" if dark else "white"
        plt.rcParams["axes.facecolor"] = "black" if dark else "white"

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

        offset = 2 * node_size + (node_border_size * 2)
        xlim_pix = [xpix[0] - offset, xpix[1] + offset]
        ylim_pix = [ypix[0] - offset, ypix[1] + offset]

        xy_coords = self.ax0.transAxes.inverted().transform(np.vstack([xlim_pix, ylim_pix]).T)
        xlim, ylim = xy_coords.T

        return xlim, ylim

    def init_fig(self):
        text_args = {
            "x": 0.5,
            "y": 1.05,
            "size": plt.rcParams["axes.titlesize"],
            "ha": "center",
            "color": "white" if self.is_dark else TITLE_GREY
        }
        ax0_title = self.ax0.text(s="Input Graph", transform=self.ax0.transAxes, **text_args)
        ax1_title = self.ax1.text(s="Modularity (Q)", transform=self.ax1.transAxes, **text_args)

        self.ax1.set_xlim([0.0, self.x[-1]])
        self.ax1.set_ylim([0.0, max(self.y)])

        for spine in self.ax0.spines.values():
            spine.set_visible(False)

        for spine in self.ax1.spines.values():
            spine.set_color(GREY if self.is_dark else LIGHT_GREY)

        self.ax1.xaxis.label.set_color(GREY if self.is_dark else LIGHT_GREY)
        self.ax1.yaxis.label.set_color(GREY if self.is_dark else LIGHT_GREY)
        self.ax1.tick_params(axis="x", which="both", bottom=False, top=False)
        self.ax1.tick_params(axis="y", colors=GREY if self.is_dark else LIGHT_GREY)
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        plt.tight_layout(pad=3.0)

        num_nodes = self.G.number_of_nodes()
        node_size = 10200 / num_nodes
        linewidths = 34 / num_nodes

        xlim, ylim = self._calculate_axes_limits(node_size, linewidths)
        self.ax0.set_xlim(xlim)
        self.ax0.set_ylim(ylim)

        self.artists = Artists(
            nx.draw_networkx_nodes(
                self.G,
                pos=self.interpolated_frames[0]["pos"],
                node_color=self.interpolated_frames[0]["C"],
                linewidths=linewidths,
                ax=self.ax0,
                cmap=cm.jet
            ),
            nx.draw_networkx_edges(
                self.G,
                pos=self.interpolated_frames[0]["pos"],
                edge_color=GREY if self.is_dark else LIGHT_GREY,
                ax=self.ax0,
                width=linewidths
            ),
            self.ax1.plot([], [], color="white" if self.is_dark else DARK_GREY)[0],
            ax0_title,
            ax1_title
        )
        self.artists.network_nodes.set_edgecolor("w")

        return self.artists

    def frame_iter(self):
        num_frames = len(self.interpolated_frames)

        # First frame (input graph)
        for _ in range(int(0.15 * num_frames)):
            yield 0

        for i in range(1, num_frames - 1):
            yield i

        # Last frame (partitioned graph)
        for _ in range(int(0.15 * num_frames)):
            yield num_frames - 1

    def update(self, i):
        if not i:
            return self.artists

        Q = self.interpolated_frames[i]["Q"]
        partition = self.interpolated_frames[i]["C"]
        pos = self.interpolated_frames[i]["pos"]
        index = self.interpolated_frames[i]["index"]

        self.artists.ax0_title.set_text(f"Iteration #{index}")
        self.artists.ax1_title.set_text(f"Modularity (Q) = {Q}")

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
        fps = int(len(self.interpolated_frames) / duration)
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frame_iter,
            init_func=self.init_fig,
            interval=1000 / fps if not filename else 200,
            save_count=len(list(self.frame_iter())),
            blit=True
        )

        if not filename:
            plt.show()
        else:
            anim.save(
                filename,
                fps=fps,
                dpi=dpi
            )

        return anim


def louvain_animation(adj_matrix, frames, dark=False, duration=15, filename=None, dpi=None, seed=2):
    anim = Animation(adj_matrix, frames, seed, dark)
    return anim.show(duration, filename, dpi)
