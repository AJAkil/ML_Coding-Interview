"""
Practice scaffold: ID3-style binary decision tree from scratch using NumPy.

Concepts to implement:
  - Shannon entropy
  - Information gain
  - Best split search (brute-force over all features and thresholds)
  - Recursive tree building with stopping conditions
  - Single-sample and batch prediction by tree traversal
"""

import numpy as np
from collections import Counter
from typing import Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ── tree node ─────────────────────────────────────────────────────────────────

class Node:
    """A node in the decision tree."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[int] = None,
    ) -> None:
        self.feature   = feature    # feature index to split on (None if leaf)
        self.threshold = threshold  # split threshold (None if leaf)
        self.left      = left       # subtree for feature <= threshold
        self.right     = right      # subtree for feature >  threshold
        self.value     = value      # predicted class (only set if leaf)

    def is_leaf(self) -> bool:
        return self.value is not None


# ── building blocks ───────────────────────────────────────────────────────────

def entropy(y: np.ndarray) -> float:
    """
    Compute Shannon entropy of label array y.
    H(y) = -sum_c p(c) * log2(p(c))

    Hint: use Counter to get class counts.
          Add a small epsilon inside log to avoid log(0).
    """
    # TODO: implement
    raise NotImplementedError


def information_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    IG = H(parent) - [|left|/n * H(left) + |right|/n * H(right)]

    Hint: call entropy() on each piece.
    """
    # TODO: implement
    raise NotImplementedError


def best_split(X: np.ndarray, y: np.ndarray):
    """
    Search every feature and every unique threshold value to find the
    (feature_idx, threshold) pair that maximises information gain.

    Returns (best_feature_idx, best_threshold).

    Steps:
      1. Loop over each feature column.
      2. For each unique value in that column, try it as a threshold.
      3. Split: left where X[:, feat] <= thresh, right where > thresh.
      4. Skip splits where one side is empty.
      5. Track the split with the highest IG.
    """
    # TODO: implement
    raise NotImplementedError


# ── recursive tree construction ───────────────────────────────────────────────

def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int = 0,
    max_depth: int = 10,
) -> Node:
    """
    Recursively build a decision tree.

    Stopping conditions (return a leaf node):
      - All labels in y are the same (pure node).
      - depth >= max_depth.
      - No valid split exists (best_split returns feat=None).

    Leaf value: majority class in y  →  Counter(y).most_common(1)[0][0]

    Otherwise:
      - Find best split.
      - Partition X and y using the split mask.
      - Recursively build left and right subtrees (depth+1).
      - Return an internal Node.
    """
    # TODO: implement
    raise NotImplementedError


# ── prediction ────────────────────────────────────────────────────────────────

def predict_sample(node: Node, x: np.ndarray) -> int:
    """
    Traverse the tree for a single sample x and return predicted class.
    At each internal node: go left if x[feature] <= threshold, else right.
    """
    # TODO: implement
    raise NotImplementedError


def predict(root: Node, X: np.ndarray) -> np.ndarray:
    """Return predicted labels for all rows in X."""
    # TODO: call predict_sample for every row
    raise NotImplementedError


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    root = build_tree(X_train, y_train, max_depth=5)
    preds = predict(root, X_test)
    print(f"Test accuracy: {accuracy_score(y_test, preds):.2%}")
