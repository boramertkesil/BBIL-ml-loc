import numpy as np


def rowwise_distances(X, Y, metric="euclidean"):
    """
    Rowwise distances.

    Parameters
    ----------
    X, Y : ndarray of shape (N, D)
    metric : str or callable, default "euclidean"

    Returns
    -------
    distances : ndarray of shape (N,)
    """
    if callable(metric):
        func = metric
    else:
        func = ROWWISE_DISTANCE_FUNCTIONS[metric]
    return func(X, Y)


def pairwise_distances(X, Y, metric="euclidean"):
    """
    Pairwise distances.

    Parameters
    ----------
    X : ndarray of shape (N, D)
    Y : ndarray of shape (M, D)
    metric : str or callable, default "euclidean"

    Returns
    -------
    distances : ndarray of shape (N, M)
    """
    if callable(metric):
        func = metric
    else:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    return func(X, Y)


# -------------------------
# Rowwise implementations
# -------------------------

def rowwise_euclidean_distances(X, Y):
    """Rowwise Euclidean distances."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of rows.")
    diff = X - Y
    return np.sqrt(np.sum(diff ** 2, axis=1))


def rowwise_manhattan_distances(X, Y):
    """Rowwise Manhattan (L1) distances."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of rows.")
    diff = np.abs(X - Y)
    return np.sum(diff, axis=1)


# -------------------------
# Pairwise implementations
# -------------------------

def pairwise_euclidean_distances(X, Y):
    """Pairwise Euclidean distance matrix."""
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def pairwise_manhattan_distances(X, Y):
    """Pairwise Manhattan (L1) distance matrix."""
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    return np.sum(diff, axis=2)


# -------------------------
# Mapping
# -------------------------

ROWWISE_DISTANCE_FUNCTIONS = {
    "euclidean": rowwise_euclidean_distances,
    "l2": rowwise_euclidean_distances,
    "manhattan": rowwise_manhattan_distances,
    "l1": rowwise_manhattan_distances,
}

PAIRWISE_DISTANCE_FUNCTIONS = {
    "euclidean": pairwise_euclidean_distances,
    "l2": pairwise_euclidean_distances,
    "manhattan": pairwise_manhattan_distances,
    "l1": pairwise_manhattan_distances,
}
