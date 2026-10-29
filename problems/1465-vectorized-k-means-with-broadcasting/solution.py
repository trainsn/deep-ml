import numpy as np

def pairwise_sq_dists(X, C):
    # squared norm of each point and each centroid
    x_sq = np.sum(X**2, axis=1)[:, None]      # (n,1)
    c_sq = np.sum(C**2, axis=1)[None, :]      # (1,k)
    # cross term via one matrix multiply
    cross = X @ C.T                            # (n,k)
    d = x_sq + c_sq - 2 * cross
    return np.clip(d, 0, None)

def assign_clusters(X, C):
    d = pairwise_sq_dists(X, C)
    return np.argmin(d, axis=1)

def update_centroids(X, labels, C):
    k, d = C.shape
    new_C = C.copy()
    # one-hot matrix: (n, k), each row has a single 1 at its cluster index
    one_hot = np.eye(k)[labels]
    counts = one_hot.sum(axis=0)                # (k,) points per cluster
    sums = one_hot.T @ X                        # (k,d) sum of points per cluster
    nonempty = counts > 0
    new_C[nonempty] = sums[nonempty] / counts[nonempty, None]
    return new_C

def kmeans(X, C0, max_iter=100, tol=1e-6):
    C = C0.copy()
    for n_iter in range(1, max_iter + 1):
        labels = assign_clusters(X, C)
        new_C = update_centroids(X, labels, C)
        shift = np.linalg.norm(new_C - C, axis=1).max()
        C = new_C
        if shift < tol:
            break
    labels = assign_clusters(X, C)
    return C, labels, n_iter

def inertia(X, C, labels):
    diffs = X - C[labels]
    return np.sum(diffs**2)