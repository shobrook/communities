# Standard Library
import random
from collections import defaultdict
from copy import copy

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull, Delaunay


##################
# COMMUNITY LAYOUT
##################


def _inter_community_edges(G, partition):
    edges = defaultdict(list)

    for (i, j) in G.edges():
        c_i = partition[i]
        c_j = partition[j]

        if c_i == c_j:
            continue

        edges[(c_i, c_j)].append((i, j))

    return edges


def _position_communities(G, partition, **kwargs):
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(set(partition))

    inter_community_edges = _inter_community_edges(G, partition)
    for (c_i, c_j), edges in inter_community_edges.items():
        hypergraph.add_edge(c_i, c_j, weight=len(edges))

    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # Set node positions to positions of its community
    pos = dict()
    for node, community in enumerate(partition):
        pos[node] = pos_communities[community]

    return pos


def _position_nodes(G, partition, **kwargs):
    communities = defaultdict(list)
    for node, community in enumerate(partition):
        communities[community].append(node)

    pos = dict()
    for c_i, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


# Adapted from: https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(G, partition):
    pos_communities = _position_communities(G, partition, scale=10.0)
    pos_nodes = _position_nodes(G, partition, scale=2.0)

    # Combine positions
    pos = dict()
    for node in G.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


#########
# PATCHES
#########


def _node_coordinates(nodes):
    collection = copy(nodes)
    collection.set_offset_position("data")
    return collection.get_offsets()


def _convex_hull_vertices(node_coordinates, community):
    points = np.array(node_coordinates[list(community)])
    hull = ConvexHull(points)

    x, y = points[hull.vertices, 0], points[hull.vertices, 1]
    vertices = np.column_stack((x, y))

    return vertices


# https://en.wikipedia.org/wiki/Shoelace_formula#Statement
def _convex_hull_area(vertices):
    A = 0.0
    for i in range(-1, vertices.shape[0] - 1):
        A += vertices[i][0] * (vertices[i + 1][1] - vertices[i - 1][1])

    return A / 2


# https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
def _convex_hull_centroid(vertices):
    A = _convex_hull_area(vertices)

    c_x, c_y = 0.0, 0.0
    for i in range(vertices.shape[0]):
        x_i, y_i = vertices[i]
        if i == vertices.shape[0] - 1:
            x_i1, y_i1 = vertices[0]
        else:
            x_i1, y_i1 = vertices[i + 1]

        cross = ((x_i * y_i1) - (x_i1 * y_i))

        c_x += (x_i + x_i1) * cross
        c_y += (y_i + y_i1) * cross

    return c_x / (6 * A), c_y / (6 * A)


def _scale_convex_hull(vertices, offset):
    c_x, c_y = _convex_hull_centroid(vertices)
    for i, vertex in enumerate(vertices):
        v_x, v_y = vertex

        if v_x > c_x:
            vertices[i][0] += offset
        else:
            vertices[i][0] -= offset
        if v_y > c_y:
            vertices[i][1] += offset
        else:
            vertices[i][1] -= offset

    return vertices


def _community_patch(vertices):
    V = _scale_convex_hull(vertices, 1) # TODO: Make offset dynamic
    tck, u = splprep(V.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    path = Path(np.column_stack((x_new, y_new)))
    patch = PathPatch(path, alpha=0.50, linewidth=0.0)
    return patch


def draw_community_patches(nodes, communities, axes):
    node_coordinates = _node_coordinates(nodes)
    vertex_sets = []
    for c_i, community in enumerate(communities):
        vertices = _convex_hull_vertices(node_coordinates, community)
        patch = _community_patch(vertices)
        patch.set_facecolor(nodes.to_rgba(c_i))

        axes.add_patch(patch)
        vertex_sets.append(patch.get_path().vertices)

    _vertices = np.concatenate(vertex_sets)
    xlim = [_vertices[:, 0].min(), _vertices[:, 0].max()]
    ylim = [_vertices[:, 1].min(), _vertices[: ,1].max()]

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


##################
# DRAW COMMUNITIES
##################


def draw_communities(adj_matrix, communities, dark=False, filename=None, dpi=None, seed=1):
    np.random.seed(seed)
    random.seed(seed)

    G = nx.from_numpy_matrix(adj_matrix)
    partition = [0 for _ in range(G.number_of_nodes())]
    for c_i, nodes in enumerate(communities):
        for i in nodes:
            partition[i] = c_i

    plt.rcParams["figure.facecolor"] = "black" if dark else "white"
    plt.rcParams["axes.facecolor"] = "black" if dark else "white"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    node_size = 10200 / G.number_of_nodes()
    linewidths = 34 / G.number_of_nodes()

    pos = community_layout(G, partition)
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=partition,
        linewidths=linewidths,
        cmap=cm.jet,
        ax=ax
    )
    nodes.set_edgecolor("w")
    edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        edge_color=(1.0, 1.0, 1.0, 0.75) if dark else (0.6, 0.6, 0.6, 1.0),
        width=linewidths,
        ax=ax
    )
    draw_community_patches(nodes, communities, ax)

    if not filename:
        plt.show()
    else:
        plt.savefig(filename, dpi=dpi)

    return ax
