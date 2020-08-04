# communities

`communities` is a library for detecting [community structure](https://en.wikipedia.org/wiki/Community_structure) in graphs. It comes with the following algorithms:

- Louvain method
- Girvan-Newman algorithm
- Hierarchical clustering
- Spectral clustering (TODO)
- Karger's algorithm (TODO)
- Bron-Kerbosch algorithm (TODO)

## Installation

`communities` can be installed with `pip`:

```bash
$ pip install communities
```

## Getting Started

Each algorithm expects an adjacency matrix representing an undirected graph. This matrix should be a 2D `numpy` array. Once you have this, just import the algorithm you want to use from `communities.algorithms` and plug in the matrix.

TODO: Add picture of graph

```python
import numpy as np
from communities.algorithms import louvain_method

adj_matrix = np.array([[0, 1, 1, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1, 0]])
communities = louvain_method(adj_matrix)
# >>> [{0, 1, 2}, {3, 4, 5}]
```

The output of each algorithm is a list of communities, where each community is a set of nodes.

## API

### communities.algorithms

#### `louvain_method(adj_matrix : numpy.ndarray, n : int = None) -> list`

Implementation of the Louvain method, from _[Fast unfolding of communities in large networks](https://arxiv.org/pdf/0803.0476.pdf)_. This algorithm does a greedy search for the communities that maximize the modularity of the graph. A graph is said to be modular if it has a high density of intra-community edges and a low density of inter-community edges. Formally, modularity is defined as:

<p align="left"><img src="modularity.png" width="275px" /></p>

where

- _A<sub>ij</sub>_ is the weight of the edge between nodes _i_ and _j_
- _k<sub>i</sub>_ and _k<sub>j</sub>_ are the sum of the weights of the edges attached to nodes _i_ and _j_, respectively
- _m_ is the sum of all of the edge weights in the graph
- _c<sub>i</sub>_ and _c<sub>j</sub>_ are the communities of the nodes
- _δ_ is the Kronecker delta function (_δ(x, y) = 1_ if _x = y_, _0_ otherwise)

Louvain's method runs in _O(nᆞlog<sup>2</sup>n)_ time, where _n_ is the number of nodes in the graph.

**Parameters:**

- `adj_matrix` _(numpy.ndarray)_: Adjacency matrix representation of your graph
- `n` _(int or None, optional (default=None))_: Terminates the search once this number of communities is detected; if `None`, then the algorithm will behave normally and terminate once modularity is maximized

**Example Usage:**

```python
from communities.algorithms import louvain_method

adj_matrix = [...]
communities = louvain_method(adj_matrix)
```

#### `girvan_newman(adj_matrix : numpy.ndarray, n : int = None) -> list`

Implementation of the Girvan-Newman algorithm, from _[Community structure in social and biological networks](https://www.pnas.org/content/99/12/7821)_. This algorithm iteratively removes edges to create more [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)). Each component is considered a community, and the algorithm stops removing edges when no more gains in modularity can be made. Edges with the highest betweenness centralities are removed.<!-- These are the edges that lie between many pairs of nodes.--> Formally, edge betweenness centrality is defined as:

<p align="left"><img src="edge_betweenness_centrality.png" width="175px" /></p>

where

- _σ(i,j)_ is the number of shortest paths from node _i_ to _j_
- _σ(i,j|e)_ is the number of shortest paths that pass through edge _e_

The Girvan-Newman algorithm runs in _O(m<sup>2</sup>n)_ time, where _m_ is the number of edges in the graph and _n_ is the number of nodes.

**Parameters:**

- `adj_matrix` _(numpy.ndarray)_: Adjacency matrix representation of your graph
    - If your graph is weighted, then the weights need to be transformed into distances, since that's how they'll be interpreted when searching for shortest paths. One way to do this is to simply take the inverse of each weight.
- `n` _(int or None, optional (default=None))_: Terminates the search once this number of communities is detected; if `None`, then the algorithm will behave normally and terminate once modularity is maximized

**Example Usage:**

```python
from communities.algorithms import girvan_newman

adj_matrix = [...]
communities = girvan_newman(adj_matrix)
```

#### `hierarchical_clustering(adj_matrix : numpy.ndarray, metric : str = "cosine", linkage : str = "single", n : int = None) -> list`

Implementation of a bottom-up, hierarchical clustering algorithm. Each node starts in its own community. Then, the most similar pairs of communities are merged as the hierarchy is built up. Communities are merged until no further gains in modularity can be made.

There are multiple schemes for measuring the similarity between two communities, _C<sub>1</sub>_ and _C<sub>1</sub>_:
- **Single-linkage:** _min({sim(i, j) | i∊C<sub>1</sub>, j∊C<sub>2</sub>})_
- **Complete-linkage:** _max({sim(i, j) | i∊C<sub>1</sub>, j∊C<sub>2</sub>})_
- **Mean-linkage:** _mean({sim(i, j) | i∊C<sub>1</sub>, j∊C<sub>2</sub>})_

where _sim(i, j)_ is the similarity between nodes _i_ and _j_, defined as either the cosine similarity or inverse Euclidean distance between their row vectors in the adjacency matrix, _A<sub>i</sub>_ and _A<sub>j</sub>_.

This algorithm runs in _O(n<sup>3</sup>)_ time, where _n_ is the number of nodes in the graph.

**Parameters:**

- `adj_matrix` _(numpy.ndarray)_: Adjacency matrix representation of your graph
- `metric` _(str, optional (default="cosine"))_: Scheme for measuring node similarity; options are "cosine", for cosine similarity, or "euclidean", for inverse Euclidean distance
- `linkage` _(str, optional (default="single"))_: Scheme for measuring community similarity; options are "single", "complete", and "mean"
- `n` _(int or None, optional (default=None))_: Terminates the search once this number of communities is detected; if `None`, then the algorithm will behave normally and terminate once modularity is maximized

**Example Usage:**

```python
from communities.algorithms import hierarchical_clustering

adj_matrix = [...]
communities = hierarchical_clustering(adj_matrix, metric="euclidean", linkage="complete")
```

### communities.utilities

#### `intercommunity_graph(adj_matrix : numpy.ndarray, communities : list, linkage : Callable = sum) -> list`

#### `modularity_matrix(adj_matrix : numpy.ndarray) -> numpy.ndarray`

#### `modularity(mod_matrix : numpy.ndarray, communities : list)

#### `binarize_matrix(adj_matrix : numpy.ndarray, threshold : float = 0.0) -> list`

This function converts a weighted graph into an unweighted graph by removing edges with weights below a given threshold.
