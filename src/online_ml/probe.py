"""Deterministic full-batch logistic regression with a text-weight prior."""

import numpy as np


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite two-dimensional array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Embeddings must have nonzero norm.")
    return values / norms


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    # This form is stable even for logits far outside the float exponential range.
    return np.exp(-np.logaddexp(0.0, -values))


class TextPriorProbe:
    def __init__(self, prior: np.ndarray, regularization: float = 0.1, steps: int = 300,
                 balanced: bool = True):
        self.prior = np.asarray(prior, dtype=np.float64).copy()
        if self.prior.ndim != 1 or not np.isfinite(self.prior).all():
            raise ValueError("Text prior must be a finite vector.")
        if not np.isfinite(regularization) or regularization <= 0 or steps < 1:
            raise ValueError("Regularization and optimizer steps must be positive.")
        self.regularization = regularization
        self.steps = steps
        self.balanced = balanced
        self.weights = self.prior.copy()
        self.bias = 0.0

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "TextPriorProbe":
        """Rebuild from the prior; label order and previous fits cannot affect the result."""
        x = unit_rows(embeddings)
        y = np.asarray(labels, dtype=np.float64)
        if x.shape[1] != len(self.prior) or y.shape != (len(x),):
            raise ValueError("Feature and label dimensions do not match.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be 0 or 1.")
        self.weights = self.prior.copy()
        self.bias = 0.0
        if not len(x):
            return self
        # Class-balanced sample weights (mean 1): each label class present
        # contributes equally, so a lopsided feedback set cannot drag the
        # intercept toward the majority class.
        classes, counts = np.unique(y, return_counts=True)
        if self.balanced:
            per_class = len(y) / (len(classes) * counts)
            w = per_class[np.searchsorted(classes, y)]
        else:
            w = np.ones_like(y)
        # Also regularize the intercept toward zero. For unit rows the augmented
        # design has squared row norm 2; with mean-one weights the Hessian is
        # still bounded by .5 + lambda, giving a safe step size.
        rate = 1.0 / (0.5 + self.regularization)
        for _ in range(self.steps):
            errors = w * (sigmoid(x @ self.weights + self.bias) - y)
            dw = x.T @ errors / len(x) + self.regularization * (self.weights - self.prior)
            db = errors.mean() + self.regularization * self.bias
            self.weights -= rate * dw
            self.bias -= rate * db
            if max(np.max(np.abs(dw)), abs(db)) < 1e-8:
                break
        return self

    def scores(self, embeddings: np.ndarray, *, baseline: bool = False) -> np.ndarray:
        x = unit_rows(embeddings)
        return sigmoid(x @ (self.prior if baseline else self.weights)
                       + (0.0 if baseline else self.bias))


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(labels)
    p = np.asarray(scores) >= 0.5
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary labels and scores.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    tp = int(np.sum((y == 1) & p))
    tn = int(np.sum((y == 0) & ~p))
    fp = int(np.sum((y == 0) & p))
    fn = int(np.sum((y == 1) & ~p))
    pos_recall = tp / (tp + fn) if tp + fn else None
    neg_recall = tn / (tn + fp) if tn + fp else None
    balanced = (pos_recall + neg_recall) / 2 if None not in (pos_recall, neg_recall) else None
    return {
        "n": len(y), "accuracy": (tp + tn) / len(y) if len(y) else None,
        "balanced_accuracy": balanced,
        "positive_recall": pos_recall, "negative_recall": neg_recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
