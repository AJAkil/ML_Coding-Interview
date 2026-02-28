import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    PCA from scratch: standardize, covariance, eigendecomposition, top-k components.

    Sign convention: flip each eigenvector so its first non-zero element is positive.
    Uses np.linalg.eigh (symmetric matrices → stable and real eigenvalues).

    Args:
        data: (n_samples, n_features)
        k: number of principal components

    Returns:
        components: (n_features, k), each column is a PC (eigenvector).
    """
    # data: (n_samples, n_features)
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # data_standardized: (n_samples, n_features)

    data_cov = np.cov(data_standardized, rowvar=False)
    # data_cov: (n_features, n_features)

    eigenvalues, eigenvectors = np.linalg.eigh(data_cov)
    # eigenvalues:  (n_features,)  — eigh returns them in ascending order (smallest first)
    # eigenvectors: (n_features, n_features), column i = eigenvector for eigenvalues[i]

    # ── Reorder by LARGEST eigenvalues first ─────────────────────────────────
    # argsort(eigenvalues) = indices that would SORT ascending: [smallest_idx, ..., largest_idx]
    # [::-1] reverses → [largest_idx, ..., smallest_idx]
    idx = np.argsort(eigenvalues)[::-1]
    # idx: (n_features,) — e.g. [2, 0, 1] means "2nd eigenvalue is largest, then 0th, then 1st"

    eigenvalues = eigenvalues[idx]
    # eigenvalues: (n_features,) — now sorted descending (largest first)

    eigenvectors = eigenvectors[:, idx]
    # eigenvectors: (n_features, n_features) — column 0 = top PC, column 1 = 2nd, ...

    components = eigenvectors[:, :k]
    # components: (n_features, k) — top k principal components

    # Sign convention: first non-zero element of each PC should be positive
    for j in range(components.shape[1]):
        mask = np.abs(components[:, j]) > 1e-10
        if np.any(mask):
            index = np.where(mask)[0][0]
            if components[index, j] < 0:
                components[:, j] *= -1

    return np.round(components, 4)