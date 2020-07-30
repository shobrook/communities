# communities

`communities` is a collection of community detection algorithms for graphs. It provides the following algorithms:

1. Louvain's modularity
2. Girvan-Newman
3. Hierarchical clustering (TODO)
4. Minimum cut (TODO)

## Installation

`communities` can be installed with `pip`:

```bash
$ pip install communities
```

## Usage

Each algorithm expects an adjacency matrix representing an undirected graph. This matrix can either be left-triangular or symmetric. To get started, just import the algorithm you want to use from `communities.algorithms`, like so:

```python
from communities.algorithms import girvan_newman

adj_matrix = [...]
communities = girvan_newman(adj_matrix)
```

The output of each algorithm is a list of communities, where each community is a set of nodes.

TODO: Implement `communities.utilities` and `hierarchical_clustering` + `min_cut`