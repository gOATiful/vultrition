from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.cluster import HDBSCAN
except ImportError as exc:
    raise ImportError(
        "HDBSCAN requires scikit-learn with HDBSCAN support. "
        "Install or upgrade with: pip install -U scikit-learn"
    ) from exc


@dataclass
class HDBSCANClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    clusters: dict[int, list[int]]
    cluster_sizes: dict[int, int]
    noise_indices: list[int]
    num_samples: int
    num_clusters: int
    noise_count: int
    noise_ratio: float


@dataclass
class ClusterDiversity:
    diversity_score: float
    simpson_diversity: float
    effective_num_groups: float
    num_samples: int
    num_hdbscan_clusters: int
    num_structural_groups: int
    noise_count: int
    noise_ratio: float
    largest_group_size: int
    largest_group_ratio: float
    structural_group_sizes: dict[int, int]


@dataclass
class EmbeddingClusterDiversityResult:
    cluster_result: HDBSCANClusterResult
    diversity: ClusterDiversity


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (num_samples, embedding_dim), "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("Cannot cluster empty embeddings")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    return embeddings.astype(np.float32, copy=False)


def _l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def cluster_code_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = 1,
    normalize_embeddings: bool = True,
    metric: str = "euclidean",
    n_jobs: int = -1,
) -> HDBSCANClusterResult:
    """
    Cluster code embeddings using HDBSCAN.

    Args:
        embeddings:
            Array with shape (num_samples, embedding_dim), e.g. (7578, 1536).

        min_cluster_size:
            Minimum number of samples required for a group to be considered
            a cluster.

        min_samples:
            Controls how conservative HDBSCAN is.

            min_samples=1 is permissive and useful for diversity analysis.
            Larger values produce fewer clusters and more noise.

        normalize_embeddings:
            If True, L2-normalizes embeddings before clustering.

            This is recommended for embedding vectors because it makes
            Euclidean distance behave similarly to cosine distance.

        metric:
            Distance metric for HDBSCAN. For normalized embeddings,
            "euclidean" is a good default.

        n_jobs:
            Number of CPU jobs. -1 uses all available CPUs.

    Returns:
        HDBSCANClusterResult
    """

    embeddings = _validate_embeddings(embeddings)

    if normalize_embeddings:
        embeddings = _l2_normalize_embeddings(embeddings)

    num_samples = embeddings.shape[0]

    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2 for meaningful clustering")

    if num_samples < min_cluster_size:
        raise ValueError(
            f"Need at least min_cluster_size={min_cluster_size} samples, "
            f"but got only {num_samples}"
        )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        n_jobs=n_jobs,
    )

    labels = clusterer.fit_predict(embeddings)

    probabilities = getattr(clusterer, "probabilities_", None)
    if probabilities is None:
        probabilities = np.ones(num_samples, dtype=np.float32)
    else:
        probabilities = np.asarray(probabilities, dtype=np.float32)

    clusters: dict[int, list[int]] = defaultdict(list)

    for index, label in enumerate(labels):
        clusters[int(label)].append(index)

    cluster_sizes = dict(Counter(int(label) for label in labels))

    noise_indices = clusters.get(-1, [])
    noise_count = len(noise_indices)
    noise_ratio = noise_count / num_samples

    real_cluster_labels = [label for label in cluster_sizes if label != -1]

    return HDBSCANClusterResult(
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=probabilities,
        clusters=dict(clusters),
        cluster_sizes=cluster_sizes,
        noise_indices=noise_indices,
        num_samples=num_samples,
        num_clusters=len(real_cluster_labels),
        noise_count=noise_count,
        noise_ratio=noise_ratio,
    )


def compute_cluster_diversity_from_labels(
    labels: np.ndarray,
    noise_as_singletons: bool = True,
) -> ClusterDiversity:
    """
    Compute structural diversity from HDBSCAN labels.

    Args:
        labels:
            HDBSCAN labels with shape (num_samples,).
            Label -1 means noise / unclustered.

        noise_as_singletons:
            If True, every HDBSCAN noise point is treated as its own
            structural group.

            This is recommended for diversity analysis because HDBSCAN noise
            points are structurally not part of any dense similarity group.

    Returns:
        ClusterDiversity

    Main metric:
        diversity_score in [0, 1]

        0.0 means all samples are in one structural group.
        1.0 means every sample is structurally unique.
    """

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(
            f"Expected labels with shape (num_samples,), got {labels.shape}"
        )

    num_samples = len(labels)

    if num_samples == 0:
        raise ValueError("Cannot compute diversity for zero labels")

    structural_labels: list[int] = []
    next_noise_label = -2

    for label in labels:
        label = int(label)

        if label == -1 and noise_as_singletons:
            structural_labels.append(next_noise_label)
            next_noise_label -= 1
        else:
            structural_labels.append(label)

    group_sizes = dict(Counter(structural_labels))
    sizes = np.array(list(group_sizes.values()), dtype=np.float64)
    probabilities = sizes / num_samples

    entropy = -np.sum(probabilities * np.log(probabilities))

    if num_samples > 1:
        max_entropy = np.log(num_samples)
        diversity_score = float(entropy / max_entropy)
    else:
        diversity_score = 0.0

    simpson = 1.0 - float(np.sum(probabilities**2))

    if num_samples > 1:
        max_simpson = 1.0 - 1.0 / num_samples
        simpson_diversity = float(simpson / max_simpson)
    else:
        simpson_diversity = 0.0

    effective_num_groups = float(np.exp(entropy))

    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / num_samples

    hdbscan_cluster_labels = set(int(label) for label in labels if int(label) != -1)

    largest_group_size = int(np.max(sizes))
    largest_group_ratio = largest_group_size / num_samples

    return ClusterDiversity(
        diversity_score=diversity_score,
        simpson_diversity=simpson_diversity,
        effective_num_groups=effective_num_groups,
        num_samples=num_samples,
        num_hdbscan_clusters=len(hdbscan_cluster_labels),
        num_structural_groups=len(group_sizes),
        noise_count=noise_count,
        noise_ratio=noise_ratio,
        largest_group_size=largest_group_size,
        largest_group_ratio=largest_group_ratio,
        structural_group_sizes={
            int(group): int(size)
            for group, size in sorted(group_sizes.items(), key=lambda x: x[0])
        },
    )


def compute_embedding_cluster_diversity(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = 1,
    normalize_embeddings: bool = True,
    noise_as_singletons: bool = True,
    metric: str = "euclidean",
    n_jobs: int = -1,
) -> EmbeddingClusterDiversityResult:
    """
    Full pipeline:
        embeddings -> HDBSCAN clusters -> diversity metrics

    Args:
        embeddings:
            Array with shape (num_samples, embedding_dim).

        min_cluster_size:
            Minimum cluster size for HDBSCAN.

        min_samples:
            HDBSCAN conservativeness parameter.

        normalize_embeddings:
            Whether to L2-normalize embeddings before clustering.

        noise_as_singletons:
            Whether each noise point should count as its own structural group
            for diversity.

        metric:
            HDBSCAN distance metric.

        n_jobs:
            Number of CPU jobs. -1 uses all available CPUs.

    Returns:
        EmbeddingClusterDiversityResult
    """

    cluster_result = cluster_code_embeddings_hdbscan(
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        normalize_embeddings=normalize_embeddings,
        metric=metric,
        n_jobs=n_jobs,
    )

    diversity = compute_cluster_diversity_from_labels(
        labels=cluster_result.labels,
        noise_as_singletons=noise_as_singletons,
    )

    return EmbeddingClusterDiversityResult(
        cluster_result=cluster_result,
        diversity=diversity,
    )


def cluster_diversity_to_dict(
    result: EmbeddingClusterDiversityResult,
) -> dict[str, Any]:
    """
    Convert result dataclasses into a JSON-serializable dictionary.
    """

    cluster_result = result.cluster_result
    diversity = result.diversity

    return {
        "clustering": {
            "num_samples": cluster_result.num_samples,
            "num_clusters": cluster_result.num_clusters,
            "noise_count": cluster_result.noise_count,
            "noise_ratio": cluster_result.noise_ratio,
            "cluster_sizes": {
                str(label): size
                for label, size in cluster_result.cluster_sizes.items()
            },
        },
        "diversity": {
            "diversity_score": diversity.diversity_score,
            "simpson_diversity": diversity.simpson_diversity,
            "effective_num_groups": diversity.effective_num_groups,
            "num_samples": diversity.num_samples,
            "num_hdbscan_clusters": diversity.num_hdbscan_clusters,
            "num_structural_groups": diversity.num_structural_groups,
            "noise_count": diversity.noise_count,
            "noise_ratio": diversity.noise_ratio,
            "largest_group_size": diversity.largest_group_size,
            "largest_group_ratio": diversity.largest_group_ratio,
            "structural_group_sizes": {
                str(label): size
                for label, size in diversity.structural_group_sizes.items()
            },
        },
    }


def print_cluster_diversity_summary(
    result: EmbeddingClusterDiversityResult,
) -> None:
    """
    Print a compact human-readable summary.
    """

    cluster_result = result.cluster_result
    diversity = result.diversity

    print("Cluster diversity summary")
    print("-" * 80)
    print(f"Samples:                  {cluster_result.num_samples}")
    print(f"HDBSCAN clusters:         {cluster_result.num_clusters}")
    print(f"Structural groups:        {diversity.num_structural_groups}")
    print(f"Noise samples:            {cluster_result.noise_count}")
    print(f"Noise ratio:              {cluster_result.noise_ratio:.2%}")
    print(f"Largest group size:       {diversity.largest_group_size}")
    print(f"Largest group ratio:      {diversity.largest_group_ratio:.2%}")
    print(f"Diversity score:          {diversity.diversity_score:.4f}")
    print(f"Simpson diversity:        {diversity.simpson_diversity:.4f}")
    print(f"Effective groups:         {diversity.effective_num_groups:.2f}")