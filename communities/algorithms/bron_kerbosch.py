# Third Party
import numpy as np


######
# MAIN
######


def bron_kerbosch(adj_matrix : np.ndarray, pivot : bool = False) -> list:
    maximal_cliques = []

    def N(v):
        return {i for i, weight in enumerate(adj_matrix[v]) if weight}

    def _bron_kerbosch(R, P, X):
        if not P and not X:
            maximal_cliques.append(R)
        else:
            if pivot:
                u = max(P | X, key=lambda i: len(N(i)))
                _P = P.copy() - N(u)
            else:
                _P = P.copy()

            for v in _P:
                _bron_kerbosch(R | {v}, P & N(v), X & N(v))
                P.remove(v)
                X.add(v)

    R, P, X = set(), set(range(len(adj_matrix))), set()
    _bron_kerbosch(R, P, X)

    return maximal_cliques
