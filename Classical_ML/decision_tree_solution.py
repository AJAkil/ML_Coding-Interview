"""
Solution: ID3-style binary decision tree from scratch using NumPy.
"""

import numpy as np
from collections import Counter
from typing import Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ── tree node ─────────────────────────────────────────────────────────────────

class Node:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[int] = None,
    ) -> None:
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self) -> bool:
        return self.value is not None


# ── building blocks ───────────────────────────────────────────────────────────

def entropy(y: np.ndarray) -> float:
    """Shannon entropy: H(y) = -sum_c p_c * log2(p_c)."""
    n = len(y)
    counts = Counter(y)
    # 1e-9 guards against log(0) when a class has zero samples
    return -sum((c / n) * np.log2(c / n + 1e-9) for c in counts.values())


def information_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """IG = H(parent) - weighted average entropy of children."""
    n = len(y)
    weighted_child_entropy = (
        (len(y_left)  / n) * entropy(y_left) +
        (len(y_right) / n) * entropy(y_right)
    )
    return entropy(y) - weighted_child_entropy


def best_split(X: np.ndarray, y: np.ndarray):
    """
    Brute-force search over all (feature, threshold) pairs.
    Returns (best_feature_idx, best_threshold) or (None, None) if no gain.
    """
    best_gain  = -1.0
    best_feat  = None
    best_thresh = None

    for feat in range(X.shape[1]):
        col = X[:, feat]
        for thresh in np.unique(col):
            mask = col <= thresh
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue    # skip degenerate splits
            gain = information_gain(y, y[mask], y[~mask])
            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

    return best_feat, best_thresh


# ── recursive tree construction ───────────────────────────────────────────────

def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int = 0,
    max_depth: int = 10,
) -> Node:
    # Stopping conditions → leaf node
    if len(set(y)) == 1 or depth >= max_depth or len(y) == 0:
        return Node(value=Counter(y).most_common(1)[0][0])

    feat, thresh = best_split(X, y)
    if feat is None:    # no split improves entropy
        return Node(value=Counter(y).most_common(1)[0][0])

    mask  = X[:, feat] <= thresh
    left  = build_tree(X[mask],  y[mask],  depth + 1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth + 1, max_depth)
    return Node(feature=feat, threshold=thresh, left=left, right=right)


# ── prediction ────────────────────────────────────────────────────────────────

def predict_sample(node: Node, x: np.ndarray) -> int:
    """Walk the tree to a leaf for a single sample."""
    if node.is_leaf():
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_sample(node.left, x)
    return predict_sample(node.right, x)


def predict(root: Node, X: np.ndarray) -> np.ndarray:
    return np.array([predict_sample(root, x) for x in X])


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    root = build_tree(X_train, y_train, max_depth=5)
    preds = predict(root, X_test)
    print(f"Test accuracy: {accuracy_score(y_test, preds):.2%}")
