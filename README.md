# communities

`communities` is a collection of community detection algorithms for graphs. It provides the following algorithms:

1. [Louvain's modularity](https://en.wikipedia.org/wiki/Louvain_modularity)
2. [Girvan-Newman](https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm)
3. Hierarchical clustering (TODO)
4. Minimum cut (TODO)

## Installation

`communities` can be installed with `pip`:

```bash
$ pip install communities
```

## Getting Started

Each algorithm expects an adjacency matrix representing an undirected graph. This matrix can either be left-triangular or symmetric. To get started, just import the algorithm you want to use from `communities.algorithms`, like so:

```python
from communities.algorithms import girvan_newman

adj_matrix = [...]
communities = girvan_newman(adj_matrix)
```

The output of each algorithm is a list of communities, where each community is a set of nodes.

## API

### `communities.algorithms`

#### `louvain_modularity(adj_matrix : list, size : int = None) -> list`

#### `girvan_newman(adj_matrix : list, size : int = None) -> list`

### `communities.utilities`

#### `is_left_triangular(adj_matrix : list) -> bool`

#### `symmetrize_matrix(adj_matrix : list) -> list

#### `binarize_matrix(adj_matrix : list, threshold : float = 0.0) -> list

#### `create_intercommunity_graph(adj_matrix : list, communities : list, aggr : Callable = sum) -> list