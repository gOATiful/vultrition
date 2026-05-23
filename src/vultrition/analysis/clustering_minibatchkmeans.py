from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


@dataclass
class MiniBatchKMeansUniqueness:
    k: int
    num_samples: int

    entropy: float
    effective_num_groups: float

    uniqueness_score: float
    balance_score: float
    resolution_adjusted_uniqueness: float

    largest_group_size: int
    largest_group_ratio: float

    inertia: float
    num_nonempty_clusters: int
    nonempty_cluster_ratio: float

    cluster_sizes: dict[int, int]


@dataclass
class MiniBatchKEntropySweepStep:
    k: int

    entropy: float
    entropy_delta: float | None
    entropy_relative_delta: float | None

    effective_num_groups: float
    uniqueness_score: float
    balance_score: float
    resolution_adjusted_uniqueness: float

    num_nonempty_clusters: int
    nonempty_cluster_ratio: float
    largest_group_ratio: float

    inertia: float
    stable: bool


@dataclass
class MiniBatchKEntropySweepResult:
    final_result: MiniBatchKMeansUniqueness
    final_k: int
    stopped_due_to_stability: bool
    steps: list[MiniBatchKEntropySweepStep]


def compute_minibatch_kmeans_uniqueness(
    embeddings: np.ndarray,
    *,
    k: int,
    batch_size: int | None = None,
    random_state: int = 42,
    normalize_embeddings: bool = True,
    max_iter: int = 300,
    n_init: int = 5,
    reassignment_ratio: float = 0.01,
) -> MiniBatchKMeansUniqueness:
    """
    Compute MiniBatchKMeans-based uniqueness without PCA.

    Important:
        This function normalizes embeddings before clustering.

    Metrics:
        entropy:
            Raw Shannon entropy over cluster sizes.

        effective_num_groups:
            exp(entropy). Interpretable as the entropy-equivalent number
            of equally sized groups.

        balance_score:
            entropy / log(num_nonempty_clusters).
            Measures how evenly samples are distributed across used clusters.

        resolution_adjusted_uniqueness:
            effective_num_groups / min(k, num_samples).
            Measures how much of the requested k-resolution is effectively used.

        uniqueness_score:
            Alias for resolution_adjusted_uniqueness.
    """

    X = np.asarray(embeddings, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (num_samples, embedding_dim), "
            f"got {X.shape}"
        )

    if len(X) == 0:
        raise ValueError("Cannot cluster empty embeddings")

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    if k < 2:
        raise ValueError("k must be >= 2")

    if k > len(X):
        raise ValueError(f"k={k} cannot be larger than num_samples={len(X)}")

    if normalize_embeddings:
        X = normalize(X, norm="l2", axis=1)

    if batch_size is None:
        batch_size = max(8192, 10 * k)

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        reassignment_ratio=reassignment_ratio,
    )

    labels = model.fit_predict(X)

    counts = Counter(int(label) for label in labels)
    sizes = np.asarray(list(counts.values()), dtype=np.float64)
    probabilities = sizes / len(labels)

    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    effective_num_groups = float(math.exp(entropy))

    num_nonempty_clusters = len(counts)
    nonempty_cluster_ratio = num_nonempty_clusters / k

    if num_nonempty_clusters > 1:
        balance_score = entropy / math.log(num_nonempty_clusters)
    else:
        balance_score = 0.0

    max_possible_groups = min(k, len(labels))
    resolution_adjusted_uniqueness = effective_num_groups / max_possible_groups

    largest_group_size = int(np.max(sizes))
    largest_group_ratio = largest_group_size / len(labels)

    return MiniBatchKMeansUniqueness(
        k=k,
        num_samples=len(labels),
        entropy=float(entropy),
        effective_num_groups=effective_num_groups,
        uniqueness_score=float(resolution_adjusted_uniqueness),
        balance_score=float(balance_score),
        resolution_adjusted_uniqueness=float(resolution_adjusted_uniqueness),
        largest_group_size=largest_group_size,
        largest_group_ratio=float(largest_group_ratio),
        inertia=float(model.inertia_),
        num_nonempty_clusters=num_nonempty_clusters,
        nonempty_cluster_ratio=float(nonempty_cluster_ratio),
        cluster_sizes={
            int(label): int(size)
            for label, size in counts.items()
        },
    )
    
    
    
def sweep_minibatch_k_until_entropy_stable(
    embeddings: np.ndarray,
    *,
    start_k: int = 512,
    max_k: int = 16384,
    growth_factor: float = 2.0,
    min_entropy_delta_ratio: float = 0.005,
    patience: int = 2,
    batch_size: int | None = None,
    random_state: int = 42,
    normalize_embeddings: bool = True,
    max_iter: int = 300,
    n_init: int = 5,
    reassignment_ratio: float = 0.01,
    show_progress: bool = True,
) -> MiniBatchKEntropySweepResult:
    """
    Increase k until entropy stabilizes.

    This avoids selecting small k only because its balance-normalized uniqueness
    score is higher.

    Stability criterion:
        entropy_relative_delta < min_entropy_delta_ratio
        for `patience` consecutive k values.

    Example:
        min_entropy_delta_ratio=0.005 means stop when entropy improves by
        less than 0.5% for `patience` consecutive steps.
    """

    n = len(embeddings)

    if start_k < 2:
        raise ValueError("start_k must be >= 2")

    if max_k > n:
        max_k = n

    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be > 1.0")

    k_values: list[int] = []
    k = start_k

    while k <= max_k:
        k_values.append(int(k))

        next_k = int(round(k * growth_factor))
        if next_k <= k:
            next_k = k + 1

        k = next_k

    steps: list[MiniBatchKEntropySweepStep] = []

    previous_entropy: float | None = None
    stable_count = 0
    stopped_due_to_stability = False
    final_result: MiniBatchKMeansUniqueness | None = None

    iterator = tqdm(
        k_values,
        desc="Increasing MiniBatchKMeans k until entropy is stable",
        unit="k",
        disable=not show_progress,
    )

    for k in iterator:
        current_batch_size = batch_size
        if current_batch_size is None:
            current_batch_size = max(8192, 10 * k)

        iterator.set_postfix_str(f"k={k}, batch_size={current_batch_size}")

        result = compute_minibatch_kmeans_uniqueness(
            embeddings,
            k=k,
            batch_size=current_batch_size,
            random_state=random_state,
            normalize_embeddings=normalize_embeddings,
            max_iter=max_iter,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
        )

        entropy = result.entropy

        if previous_entropy is None:
            entropy_delta = None
            entropy_relative_delta = None
            stable = False
        else:
            entropy_delta = entropy - previous_entropy

            if abs(previous_entropy) > 1e-12:
                entropy_relative_delta = entropy_delta / previous_entropy
            else:
                entropy_relative_delta = float("inf")

            stable = abs(entropy_relative_delta) < min_entropy_delta_ratio

            if stable:
                stable_count += 1
            else:
                stable_count = 0

        step = MiniBatchKEntropySweepStep(
            k=k,
            entropy=float(result.entropy),
            entropy_delta=None if entropy_delta is None else float(entropy_delta),
            entropy_relative_delta=(
                None if entropy_relative_delta is None
                else float(entropy_relative_delta)
            ),
            effective_num_groups=result.effective_num_groups,
            uniqueness_score=result.uniqueness_score,
            balance_score=result.balance_score,
            resolution_adjusted_uniqueness=result.resolution_adjusted_uniqueness,
            num_nonempty_clusters=result.num_nonempty_clusters,
            nonempty_cluster_ratio=result.nonempty_cluster_ratio,
            largest_group_ratio=result.largest_group_ratio,
            inertia=result.inertia,
            stable=stable,
        )

        steps.append(step)
        final_result = result

        print(
            {
                "k": step.k,
                "entropy": round(step.entropy, 4),
                "entropy_delta": (
                    None
                    if step.entropy_delta is None
                    else round(step.entropy_delta, 4)
                ),
                "entropy_relative_delta": (
                    None
                    if step.entropy_relative_delta is None
                    else round(step.entropy_relative_delta, 6)
                ),
                "stable": step.stable,
                "stable_count": stable_count,
                "effective_groups": round(step.effective_num_groups, 2),
                "balance_score": round(step.balance_score, 4),
                "resolution_adjusted_uniqueness": round(
                    step.resolution_adjusted_uniqueness,
                    4,
                ),
                "nonempty_ratio": round(step.nonempty_cluster_ratio, 4),
                "largest_group_ratio": round(step.largest_group_ratio, 4),
            }
        )

        previous_entropy = entropy

        if stable_count >= patience:
            stopped_due_to_stability = True
            print()
            print(
                f"Stopping because entropy stabilized for {patience} consecutive "
                f"k values with min_entropy_delta_ratio={min_entropy_delta_ratio}"
            )
            break

    if final_result is None:
        raise RuntimeError("No MiniBatchKMeans run completed")

    return MiniBatchKEntropySweepResult(
        final_result=final_result,
        final_k=final_result.k,
        stopped_due_to_stability=stopped_due_to_stability,
        steps=steps,
    )