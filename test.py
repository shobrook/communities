from communities.algorithms import hierarchical_clustering
import pickle
import numpy as np

adj_matrix = pickle.load(open("test_graph.pkl", "rb"))
communities = hierarchical_clustering(np.array(adj_matrix), size=None, metric="cosine", linkage="mean")
print(communities)