from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


@dataclass
class ClusteringSummary:
    backend: str

    num_samples_total: int
    num_samples_clustered: int
    embedding_dim_original: int
    embedding_dim_used: int

    sampled: bool
    sample_size: int | None
    random_state: int

    pca_components: int | None
    pca_explained_variance_ratio_sum: float | None

    min_cluster_size: int
    min_samples: int | None
    metric: str

    normalize_before_pca: bool
    normalize_after_pca: bool
    noise_as_singletons: bool

    num_hdbscan_clusters: int
    num_structural_groups: int
    noise_count: int
    noise_ratio: float

    largest_group_size: int
    largest_group_ratio: float

    uniqueness_score: float
    simpson_diversity: float
    effective_num_groups: float

    top_cluster_sizes: list[dict[str, int]]


@dataclass
class ClusteringUniquenessResult:
    summary: ClusteringSummary
    labels: np.ndarray
    probabilities: np.ndarray
    sample_indices: np.ndarray


def validate_embeddings(
    embeddings: np.ndarray,
    check_finite: bool = True,
) -> np.ndarray:
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (num_samples, embedding_dim), "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("Cannot cluster an empty embedding matrix")

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding dimension cannot be zero")

    if check_finite and not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    return embeddings.astype(np.float32, copy=False)


def l2_normalize_numpy(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def choose_min_cluster_size(
    num_samples: int,
    min_cluster_size: int | None = None,
    min_cluster_size_ratio: float = 0.0003,
    lower_bound: int = 5,
) -> int:
    if min_cluster_size is not None:
        value = min_cluster_size
    else:
        value = int(round(num_samples * min_cluster_size_ratio))
        value = max(lower_bound, value)

    if value < 2:
        raise ValueError("min_cluster_size must be >= 2")

    if value > num_samples:
        raise ValueError(
            f"min_cluster_size={value} is larger than num_samples={num_samples}"
        )

    return value


def maybe_sample_embeddings(
    embeddings: np.ndarray,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optionally sample embeddings before clustering.

    For large datasets like 330k samples, CPU HDBSCAN on the full dataset may be
    impractical. Sampling gives an estimated uniqueness score.
    """

    num_samples = len(embeddings)

    if sample_size is None or sample_size >= num_samples:
        indices = np.arange(num_samples, dtype=np.int64)
        return embeddings, indices

    if sample_size < 2:
        raise ValueError("sample_size must be >= 2")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(num_samples, size=sample_size, replace=False)
    indices = np.sort(indices)

    return embeddings[indices], indices


def prepare_embeddings_for_clustering(
    embeddings: np.ndarray,
    *,
    pca_components: int | None = 100,
    normalize_before_pca: bool = True,
    normalize_after_pca: bool = True,
    show_progress: bool = True,
) -> tuple[np.ndarray, int, float | None]:
    """
    Normalize, optionally PCA-reduce, and normalize again.

    Returns:
        X:
            Prepared embeddings.

        embedding_dim_used:
            Final dimension used for HDBSCAN.

        pca_explained_variance_ratio_sum:
            Sum of PCA explained variance ratios, or None if PCA disabled.
    """

    progress = tqdm(
        total=3 + int(pca_components is not None),
        desc="Prepare embeddings",
        unit="step",
        disable=not show_progress,
    )

    try:
        progress.set_postfix_str("copy to float32")
        X = np.asarray(embeddings, dtype=np.float32)
        progress.update(1)

        progress.set_postfix_str("normalize before PCA")
        if normalize_before_pca:
            X = l2_normalize_numpy(X)
        progress.update(1)

        pca_explained = None

        if pca_components is not None:
            if pca_components >= X.shape[1]:
                raise ValueError(
                    f"pca_components={pca_components} must be smaller than "
                    f"embedding_dim={X.shape[1]}"
                )

            progress.set_postfix_str(f"PCA to {pca_components} dims")

            pca = PCA(
                n_components=pca_components,
                svd_solver="randomized",
                random_state=42,
            )

            X = pca.fit_transform(X).astype(np.float32, copy=False)
            pca_explained = float(np.sum(pca.explained_variance_ratio_))

            progress.update(1)

        progress.set_postfix_str("normalize after PCA")
        if normalize_after_pca:
            X = l2_normalize_numpy(X)
        progress.update(1)

        progress.set_postfix_str("done")
        embedding_dim_used = int(X.shape[1])
        progress.update(1)

        return X, embedding_dim_used, pca_explained

    finally:
        progress.close()


def run_hdbscan_cpu(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None = 1,
    metric: str = "cosine", #"euclidean",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run CPU HDBSCAN using scikit-learn.

    Returns:
        labels:
            HDBSCAN labels. Noise is -1.

        probabilities:
            HDBSCAN membership probabilities if available.
    """

    progress = tqdm(
        total=2,
        desc="CPU HDBSCAN",
        unit="step",
        disable=not show_progress,
    )

    try:
        progress.set_postfix_str("fit HDBSCAN")

        kwargs: dict[str, Any] = {
            "min_cluster_size": min_cluster_size,
            "metric": metric,
            "n_jobs": -1,
        }

        if min_samples is not None:
            kwargs["min_samples"] = min_samples

        clusterer = HDBSCAN(**kwargs)
        labels = clusterer.fit_predict(embeddings).astype(np.int64)

        progress.update(1)

        progress.set_postfix_str("read probabilities")

        probabilities = getattr(clusterer, "probabilities_", None)

        if probabilities is None:
            probabilities = np.ones(len(labels), dtype=np.float32)
        else:
            probabilities = np.asarray(probabilities, dtype=np.float32)

        progress.update(1)

        return labels, probabilities

    finally:
        progress.close()


def compute_uniqueness_from_labels(
    labels: np.ndarray,
    *,
    backend: str,
    num_samples_total: int,
    num_samples_clustered: int,
    embedding_dim_original: int,
    embedding_dim_used: int,
    sampled: bool,
    sample_size: int | None,
    random_state: int,
    pca_components: int | None,
    pca_explained_variance_ratio_sum: float | None,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    normalize_before_pca: bool,
    normalize_after_pca: bool,
    noise_as_singletons: bool = True,
) -> ClusteringSummary:
    """
    Compute uniqueness from HDBSCAN labels.

    HDBSCAN uses label -1 for noise.

    If noise_as_singletons=True, every noise point is treated as its own
    structural group. This makes the score measure uniqueness/non-redundancy.
    """

    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError(f"Expected labels with shape (num_samples,), got {labels.shape}")

    if len(labels) != num_samples_clustered:
        raise ValueError(
            f"Expected {num_samples_clustered} labels, got {len(labels)}"
        )

    label_counts = Counter(int(label) for label in labels)

    noise_count = int(label_counts.get(-1, 0))

    real_cluster_counts = {
        label: count
        for label, count in label_counts.items()
        if label != -1
    }

    if noise_as_singletons:
        structural_sizes = list(real_cluster_counts.values()) + [1] * noise_count
    else:
        structural_sizes = list(real_cluster_counts.values())
        if noise_count > 0:
            structural_sizes.append(noise_count)

    if not structural_sizes:
        structural_sizes = [num_samples_clustered]

    sizes = np.asarray(structural_sizes, dtype=np.float64)
    probabilities = sizes / float(num_samples_clustered)

    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    if num_samples_clustered > 1:
        uniqueness_score = entropy / math.log(num_samples_clustered)
    else:
        uniqueness_score = 0.0

    simpson_raw = 1.0 - float(np.sum(probabilities**2))

    if num_samples_clustered > 1:
        simpson_max = 1.0 - 1.0 / num_samples_clustered
        simpson_diversity = simpson_raw / simpson_max
    else:
        simpson_diversity = 0.0

    effective_num_groups = float(math.exp(entropy))

    num_hdbscan_clusters = len(real_cluster_counts)

    if noise_as_singletons:
        num_structural_groups = num_hdbscan_clusters + noise_count
    else:
        num_structural_groups = num_hdbscan_clusters + int(noise_count > 0)

    largest_group_size = int(np.max(sizes))
    largest_group_ratio = largest_group_size / float(num_samples_clustered)
    noise_ratio = noise_count / float(num_samples_clustered)

    top_cluster_sizes = [
        {"label": int(label), "size": int(size)}
        for label, size in sorted(
            label_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:25]
    ]

    return ClusteringSummary(
        backend=backend,
        num_samples_total=num_samples_total,
        num_samples_clustered=num_samples_clustered,
        embedding_dim_original=embedding_dim_original,
        embedding_dim_used=embedding_dim_used,
        sampled=sampled,
        sample_size=sample_size,
        random_state=random_state,
        pca_components=pca_components,
        pca_explained_variance_ratio_sum=pca_explained_variance_ratio_sum,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        normalize_before_pca=normalize_before_pca,
        normalize_after_pca=normalize_after_pca,
        noise_as_singletons=noise_as_singletons,
        num_hdbscan_clusters=num_hdbscan_clusters,
        num_structural_groups=num_structural_groups,
        noise_count=noise_count,
        noise_ratio=noise_ratio,
        largest_group_size=largest_group_size,
        largest_group_ratio=largest_group_ratio,
        uniqueness_score=float(uniqueness_score),
        simpson_diversity=float(simpson_diversity),
        effective_num_groups=float(effective_num_groups),
        top_cluster_sizes=top_cluster_sizes,
    )


def compute_clustering_uniqueness(
    embeddings: np.ndarray,
    *,
    pca_components: int | None = 100,
    min_cluster_size: int | None = None,
    min_cluster_size_ratio: float = 0.0003,
    min_cluster_size_lower_bound: int = 5,
    min_samples: int | None = 1,
    metric: str = "euclidean",
    normalize_before_pca: bool = True,
    normalize_after_pca: bool = True,
    noise_as_singletons: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
    check_finite: bool = True,
    show_progress: bool = True,
) -> ClusteringUniquenessResult:
    """
    CPU-only clustering uniqueness pipeline.

    Pipeline:
        embeddings
        -> optional sampling
        -> normalization
        -> optional PCA
        -> CPU HDBSCAN
        -> uniqueness score

    Args:
        embeddings:
            Array with shape (num_samples, embedding_dim).

        pca_components:
            PCA dimensions before clustering.
            Use None to disable PCA.

        min_cluster_size:
            Fixed HDBSCAN min_cluster_size.
            If None, computed as:
                max(min_cluster_size_lower_bound,
                    round(num_clustered_samples * min_cluster_size_ratio))

        min_samples:
            HDBSCAN min_samples.
            min_samples=1 is permissive and useful for uniqueness analysis.

        sample_size:
            Optional number of samples to cluster.
            Recommended for very large datasets when running CPU-only.

        noise_as_singletons:
            If True, each HDBSCAN noise point counts as its own structural group.

    Returns:
        ClusteringUniquenessResult:
            summary:
                Uniqueness metrics.

            labels:
                HDBSCAN labels for the clustered samples.

            probabilities:
                HDBSCAN probabilities for the clustered samples.

            sample_indices:
                Original row indices corresponding to labels.
    """

    embeddings = validate_embeddings(embeddings, check_finite=check_finite)

    num_samples_total, embedding_dim_original = embeddings.shape

    sampled_embeddings, sample_indices = maybe_sample_embeddings(
        embeddings,
        sample_size=sample_size,
        random_state=random_state,
    )

    num_samples_clustered = len(sampled_embeddings)
    sampled = num_samples_clustered != num_samples_total

    if not sampled and num_samples_total > 50_000:
        warnings.warn(
            "CPU HDBSCAN on more than 50k samples can be very slow or impractical. "
            "Consider setting sample_size=20000 or sample_size=50000.",
            RuntimeWarning,
        )

    chosen_min_cluster_size = choose_min_cluster_size(
        num_samples=num_samples_clustered,
        min_cluster_size=min_cluster_size,
        min_cluster_size_ratio=min_cluster_size_ratio,
        lower_bound=min_cluster_size_lower_bound,
    )

    print(f"Embedding shape total:    {embeddings.shape}")
    print(f"Clustered sample count:   {num_samples_clustered}")
    print(f"Sampled:                  {sampled}")
    print(f"Backend:                  cpu")
    print(f"min_cluster_size:         {chosen_min_cluster_size}")
    print(f"min_samples:              {min_samples}")
    print(f"pca_components:           {pca_components}")

    X, embedding_dim_used, pca_explained = prepare_embeddings_for_clustering(
        sampled_embeddings,
        pca_components=pca_components,
        normalize_before_pca=normalize_before_pca,
        normalize_after_pca=normalize_after_pca,
        show_progress=show_progress,
    )

    labels, probabilities = run_hdbscan_cpu(
        X,
        min_cluster_size=chosen_min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        show_progress=show_progress,
    )

    summary = compute_uniqueness_from_labels(
        labels=labels,
        backend="cpu",
        num_samples_total=num_samples_total,
        num_samples_clustered=num_samples_clustered,
        embedding_dim_original=embedding_dim_original,
        embedding_dim_used=embedding_dim_used,
        sampled=sampled,
        sample_size=sample_size,
        random_state=random_state,
        pca_components=pca_components,
        pca_explained_variance_ratio_sum=pca_explained,
        min_cluster_size=chosen_min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        normalize_before_pca=normalize_before_pca,
        normalize_after_pca=normalize_after_pca,
        noise_as_singletons=noise_as_singletons,
    )

    return ClusteringUniquenessResult(
        summary=summary,
        labels=labels,
        probabilities=probabilities,
        sample_indices=sample_indices,
    )


def print_clustering_summary(summary: ClusteringSummary) -> None:
    print()
    print("Cluster uniqueness summary")
    print("-" * 80)
    print(f"Backend:                  {summary.backend}")
    print(f"Samples total:            {summary.num_samples_total}")
    print(f"Samples clustered:        {summary.num_samples_clustered}")
    print(f"Sampled:                  {summary.sampled}")
    print(f"Original dim:             {summary.embedding_dim_original}")
    print(f"Used dim:                 {summary.embedding_dim_used}")
    print(f"PCA components:           {summary.pca_components}")

    if summary.pca_explained_variance_ratio_sum is not None:
        print(
            "PCA variance retained:    "
            f"{summary.pca_explained_variance_ratio_sum:.4f}"
        )

    print(f"min_cluster_size:         {summary.min_cluster_size}")
    print(f"min_samples:              {summary.min_samples}")
    print(f"HDBSCAN clusters:         {summary.num_hdbscan_clusters}")
    print(f"Structural groups:        {summary.num_structural_groups}")
    print(f"Noise samples:            {summary.noise_count}")
    print(f"Noise ratio:              {summary.noise_ratio:.2%}")
    print(f"Largest group size:       {summary.largest_group_size}")
    print(f"Largest group ratio:      {summary.largest_group_ratio:.2%}")
    print(f"Uniqueness score:         {summary.uniqueness_score:.4f}")
    print(f"Simpson diversity:        {summary.simpson_diversity:.4f}")
    print(f"Effective groups:         {summary.effective_num_groups:.2f}")

    print()
    print("Top cluster sizes including noise label -1")
    print("-" * 80)

    for item in summary.top_cluster_sizes[:10]:
        print(f"label={item['label']:<8} size={item['size']}")


def clustering_summary_to_dict(summary: ClusteringSummary) -> dict[str, Any]:
    return {
        "backend": summary.backend,
        "num_samples_total": summary.num_samples_total,
        "num_samples_clustered": summary.num_samples_clustered,
        "embedding_dim_original": summary.embedding_dim_original,
        "embedding_dim_used": summary.embedding_dim_used,
        "sampled": summary.sampled,
        "sample_size": summary.sample_size,
        "random_state": summary.random_state,
        "pca_components": summary.pca_components,
        "pca_explained_variance_ratio_sum": summary.pca_explained_variance_ratio_sum,
        "min_cluster_size": summary.min_cluster_size,
        "min_samples": summary.min_samples,
        "metric": summary.metric,
        "normalize_before_pca": summary.normalize_before_pca,
        "normalize_after_pca": summary.normalize_after_pca,
        "noise_as_singletons": summary.noise_as_singletons,
        "num_hdbscan_clusters": summary.num_hdbscan_clusters,
        "num_structural_groups": summary.num_structural_groups,
        "noise_count": summary.noise_count,
        "noise_ratio": summary.noise_ratio,
        "largest_group_size": summary.largest_group_size,
        "largest_group_ratio": summary.largest_group_ratio,
        "uniqueness_score": summary.uniqueness_score,
        "simpson_diversity": summary.simpson_diversity,
        "effective_num_groups": summary.effective_num_groups,
        "top_cluster_sizes": summary.top_cluster_sizes,
    }