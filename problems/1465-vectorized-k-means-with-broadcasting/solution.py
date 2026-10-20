import numpy as np


def pairwise_sq_dists(X, C):
    # TODO: ||x||^2 + ||c||^2 - 2 x.c via one matrix product, clipped at 0
    x2 = np.sum(X**2, axis=1).reshape(-1, 1)      # (n,1)
    c2 = np.sum(C**2, axis=1).reshape(1, -1)       # (1,k)
    cross = X @ C.T                                 # (n,k)
    d2 = x2 + c2 - 2 * cross
    return np.clip(d2, 0, None)


def assign_clusters(X, C):
    # TODO: argmin over centroids
    d2 = pairwise_sq_dists(X, C)
    return np.argmin(d2, axis=1)


def update_centroids(X, labels, C):
    # TODO: mean of each cluster's points; empty clusters keep their centroid
    k = C.shape[0]
    new_C = C.copy()
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            new_C[j] = X[mask].mean(axis=0)
        # else keep old centroid (already copied)
    return new_C


def kmeans(X, C0, max_iter=100, tol=1e-6):
    # TODO: iterate until the largest centroid movement < tol; return (C, labels, n_iter)
    C = C0.copy()
    labels = assign_clusters(X, C)
    for n_iter in range(1, max_iter + 1):
        new_C = update_centroids(X, labels, C)
        # largest centroid movement (Euclidean norm per row)
        shift = np.linalg.norm(new_C - C, axis=1).max()
        C = new_C
        labels = assign_clusters(X, C)
        if shift < tol:
            break
    return C, labels, n_iter


def inertia(X, C, labels):
    d2 = pairwise_sq_dists(X, C)
    # pick distance to each point's own assigned cluster, then sum
    return np.sum(d2[np.arange(X.shape[0]), labels])
