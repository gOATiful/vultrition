from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm


@dataclass
class DirectionalSimilarityResult:
    source_indices: np.ndarray
    best_target_indices: np.ndarray
    best_cosine_similarity: np.ndarray
    mean_top_k_similarity: np.ndarray

    source_ids: np.ndarray | None = None
    best_target_ids: np.ndarray | None = None

    top_k_target_indices: np.ndarray | None = None
    top_k_cosine_similarity: np.ndarray | None = None


@dataclass
class CrossDatasetSimilaritySummary:
    num_A: int
    num_B: int
    embedding_dim: int
    top_k: int
    chunk_size: int

    A_to_B_score: float
    B_to_A_score: float
    symmetric_similarity_score: float
    conservative_similarity_score: float
    pooled_similarity_score: float

    median_A_to_B_similarity: float
    median_B_to_A_similarity: float
    p10_A_to_B_similarity: float
    p10_B_to_A_similarity: float
    p90_A_to_B_similarity: float
    p90_B_to_A_similarity: float

    min_A_to_B_similarity: float
    min_B_to_A_similarity: float
    max_A_to_B_similarity: float
    max_B_to_A_similarity: float

    fraction_A_to_B_above_090: float
    fraction_B_to_A_above_090: float
    fraction_A_to_B_above_095: float
    fraction_B_to_A_above_095: float
    fraction_A_to_B_above_099: float
    fraction_B_to_A_above_099: float


@dataclass
class CrossDatasetSimilarityResult:
    summary: CrossDatasetSimilaritySummary
    A_to_B: DirectionalSimilarityResult | None
    B_to_A: DirectionalSimilarityResult | None


def _validate_embedding_matrix(name: str, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"{name} must have shape (num_samples, embedding_dim), got {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError(f"{name} has zero samples")

    if X.shape[1] == 0:
        raise ValueError(f"{name} has zero embedding dimensions")

    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return X.astype(np.float32, copy=False)


def _l2_normalize_numpy(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def _topk_numpy(
    similarities: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return top-k values and indices per row.

    Args:
        similarities:
            Matrix of shape (chunk_size, num_target)

        top_k:
            Number of largest values to return.

    Returns:
        topk_values:
            Shape (chunk_size, top_k)

        topk_indices:
            Shape (chunk_size, top_k)
    """

    if top_k == 1:
        indices = np.argmax(similarities, axis=1)[:, None]
        values = np.take_along_axis(similarities, indices, axis=1)
        return values.astype(np.float32), indices.astype(np.int64)

    # argpartition is faster than full sorting.
    partition_indices = np.argpartition(
        similarities,
        kth=-top_k,
        axis=1,
    )[:, -top_k:]

    partition_values = np.take_along_axis(
        similarities,
        partition_indices,
        axis=1,
    )

    # Sort top-k descending.
    order = np.argsort(-partition_values, axis=1)

    topk_indices = np.take_along_axis(partition_indices, order, axis=1)
    topk_values = np.take_along_axis(partition_values, order, axis=1)

    return topk_values.astype(np.float32), topk_indices.astype(np.int64)


def _directional_similarity_numpy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_ids: np.ndarray | None,
    target_ids: np.ndarray | None,
    top_k: int,
    chunk_size: int,
    desc: str,
    show_progress: bool,
    return_topk: bool,
) -> DirectionalSimilarityResult:
    """
    For every row in source, find top-k nearest rows in target by cosine similarity.

    Assumes source and target are already L2-normalized.
    """

    num_source = source.shape[0]
    num_target = target.shape[0]

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    effective_top_k = min(top_k, num_target)

    best_target_indices = np.empty(num_source, dtype=np.int64)
    best_cosine_similarity = np.empty(num_source, dtype=np.float32)
    mean_top_k_similarity = np.empty(num_source, dtype=np.float32)

    if return_topk:
        all_topk_indices = np.empty(
            (num_source, effective_top_k),
            dtype=np.int64,
        )
        all_topk_sims = np.empty(
            (num_source, effective_top_k),
            dtype=np.float32,
        )
    else:
        all_topk_indices = None
        all_topk_sims = None

    for start in tqdm(
        range(0, num_source, chunk_size),
        total=(num_source + chunk_size - 1) // chunk_size,
        desc=desc,
        unit="chunk",
        disable=not show_progress,
    ):
        end = min(start + chunk_size, num_source)

        source_chunk = source[start:end]

        # Since vectors are normalized, dot product == cosine similarity.
        similarities = source_chunk @ target.T

        topk_values, topk_indices = _topk_numpy(
            similarities,
            top_k=effective_top_k,
        )

        best_target_indices[start:end] = topk_indices[:, 0]
        best_cosine_similarity[start:end] = topk_values[:, 0]
        mean_top_k_similarity[start:end] = topk_values.mean(axis=1)

        if return_topk:
            all_topk_indices[start:end] = topk_indices
            all_topk_sims[start:end] = topk_values

        del similarities

    if source_ids is not None:
        source_ids_out = np.asarray(source_ids)
    else:
        source_ids_out = None

    if target_ids is not None:
        best_target_ids_out = np.asarray(target_ids)[best_target_indices]
    else:
        best_target_ids_out = None

    return DirectionalSimilarityResult(
        source_indices=np.arange(num_source, dtype=np.int64),
        best_target_indices=best_target_indices,
        best_cosine_similarity=best_cosine_similarity,
        mean_top_k_similarity=mean_top_k_similarity,
        source_ids=source_ids_out,
        best_target_ids=best_target_ids_out,
        top_k_target_indices=all_topk_indices,
        top_k_cosine_similarity=all_topk_sims,
    )


def _fraction_above(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values >= threshold))


def _make_cross_dataset_summary(
    A_to_B: DirectionalSimilarityResult,
    B_to_A: DirectionalSimilarityResult,
    *,
    num_A: int,
    num_B: int,
    embedding_dim: int,
    top_k: int,
    chunk_size: int,
) -> CrossDatasetSimilaritySummary:
    per_sample_A_to_B = A_to_B.mean_top_k_similarity
    per_sample_B_to_A = B_to_A.mean_top_k_similarity

    A_to_B_score = float(per_sample_A_to_B.mean())
    B_to_A_score = float(per_sample_B_to_A.mean())

    symmetric_similarity_score = float(
        0.5 * (A_to_B_score + B_to_A_score)
    )

    conservative_similarity_score = float(
        min(A_to_B_score, B_to_A_score)
    )

    pooled_similarity_score = float(
        (
            per_sample_A_to_B.sum()
            + per_sample_B_to_A.sum()
        )
        / (num_A + num_B)
    )

    return CrossDatasetSimilaritySummary(
        num_A=num_A,
        num_B=num_B,
        embedding_dim=embedding_dim,
        top_k=top_k,
        chunk_size=chunk_size,

        A_to_B_score=A_to_B_score,
        B_to_A_score=B_to_A_score,
        symmetric_similarity_score=symmetric_similarity_score,
        conservative_similarity_score=conservative_similarity_score,
        pooled_similarity_score=pooled_similarity_score,

        median_A_to_B_similarity=float(np.median(per_sample_A_to_B)),
        median_B_to_A_similarity=float(np.median(per_sample_B_to_A)),
        p10_A_to_B_similarity=float(np.quantile(per_sample_A_to_B, 0.10)),
        p10_B_to_A_similarity=float(np.quantile(per_sample_B_to_A, 0.10)),
        p90_A_to_B_similarity=float(np.quantile(per_sample_A_to_B, 0.90)),
        p90_B_to_A_similarity=float(np.quantile(per_sample_B_to_A, 0.90)),

        min_A_to_B_similarity=float(np.min(per_sample_A_to_B)),
        min_B_to_A_similarity=float(np.min(per_sample_B_to_A)),
        max_A_to_B_similarity=float(np.max(per_sample_A_to_B)),
        max_B_to_A_similarity=float(np.max(per_sample_B_to_A)),

        fraction_A_to_B_above_090=_fraction_above(per_sample_A_to_B, 0.90),
        fraction_B_to_A_above_090=_fraction_above(per_sample_B_to_A, 0.90),
        fraction_A_to_B_above_095=_fraction_above(per_sample_A_to_B, 0.95),
        fraction_B_to_A_above_095=_fraction_above(per_sample_B_to_A, 0.95),
        fraction_A_to_B_above_099=_fraction_above(per_sample_A_to_B, 0.99),
        fraction_B_to_A_above_099=_fraction_above(per_sample_B_to_A, 0.99),
    )


def embedding_dataset_similarity(
    A: np.ndarray,
    B: np.ndarray,
    ids_A: np.ndarray | list[Any] | None = None,
    ids_B: np.ndarray | list[Any] | None = None,
    top_k: int = 1,
    chunk_size: int = 256,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
    return_matches: bool = False,
    return_topk: bool = False,
) -> CrossDatasetSimilarityResult:
    """
    Compute exact cross-dataset cosine similarity.

    Main idea:
        For each sample in A, find its top-k nearest samples in B.
        For each sample in B, find its top-k nearest samples in A.
        Average both directions.

    This version is CPU-only and uses NumPy chunking.

    Args:
        A:
            Embeddings for dataset A, shape (num_A, embedding_dim).

        B:
            Embeddings for dataset B, shape (num_B, embedding_dim).

        ids_A:
            Optional IDs for A. Must have length num_A.

        ids_B:
            Optional IDs for B. Must have length num_B.

        top_k:
            Number of nearest neighbors to average per sample.

        chunk_size:
            Number of source rows processed at once.

            Larger is faster but uses more RAM.
            For large datasets, start with 128 or 256.

        normalize_embeddings:
            If True, L2-normalize embeddings first.

        show_progress:
            Show tqdm progress bars.

        return_matches:
            If True, return per-sample best matches.

        return_topk:
            If True, store all top-k matches.
            This can use extra memory.

    Returns:
        CrossDatasetSimilarityResult
    """

    A = _validate_embedding_matrix("A", A)
    B = _validate_embedding_matrix("B", B)

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A and B must have the same embedding dimension, "
            f"got A.shape={A.shape}, B.shape={B.shape}"
        )

    num_A, embedding_dim = A.shape
    num_B = B.shape[0]

    if ids_A is not None and len(ids_A) != num_A:
        raise ValueError(f"ids_A has length {len(ids_A)}, expected {num_A}")

    if ids_B is not None and len(ids_B) != num_B:
        raise ValueError(f"ids_B has length {len(ids_B)}, expected {num_B}")

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    if normalize_embeddings:
        print("Normalizing embeddings...")
        A = _l2_normalize_numpy(A)
        B = _l2_normalize_numpy(B)

    print(f"A shape:      {A.shape}")
    print(f"B shape:      {B.shape}")
    print(f"top_k:        {top_k}")
    print(f"chunk_size:   {chunk_size}")
    print("backend:      numpy/cpu")

    A_to_B = _directional_similarity_numpy(
        source=A,
        target=B,
        source_ids=np.asarray(ids_A) if ids_A is not None else None,
        target_ids=np.asarray(ids_B) if ids_B is not None else None,
        top_k=top_k,
        chunk_size=chunk_size,
        desc="A -> B similarity",
        show_progress=show_progress,
        return_topk=return_topk,
    )

    B_to_A = _directional_similarity_numpy(
        source=B,
        target=A,
        source_ids=np.asarray(ids_B) if ids_B is not None else None,
        target_ids=np.asarray(ids_A) if ids_A is not None else None,
        top_k=top_k,
        chunk_size=chunk_size,
        desc="B -> A similarity",
        show_progress=show_progress,
        return_topk=return_topk,
    )

    summary = _make_cross_dataset_summary(
        A_to_B=A_to_B,
        B_to_A=B_to_A,
        num_A=num_A,
        num_B=num_B,
        embedding_dim=embedding_dim,
        top_k=top_k,
        chunk_size=chunk_size,
    )

    if not return_matches:
        A_to_B_out = None
        B_to_A_out = None
    else:
        A_to_B_out = A_to_B
        B_to_A_out = B_to_A

    return CrossDatasetSimilarityResult(
        summary=summary,
        A_to_B=A_to_B_out,
        B_to_A=B_to_A_out,
    )


def print_cross_dataset_similarity_summary(
    summary: CrossDatasetSimilaritySummary,
) -> None:
    print()
    print("Cross-dataset similarity summary")
    print("-" * 80)
    print(f"A samples:                         {summary.num_A}")
    print(f"B samples:                         {summary.num_B}")
    print(f"Embedding dim:                     {summary.embedding_dim}")
    print(f"Top-k:                             {summary.top_k}")
    print(f"Chunk size:                        {summary.chunk_size}")

    print()
    print(f"A -> B score:                      {summary.A_to_B_score:.4f}")
    print(f"B -> A score:                      {summary.B_to_A_score:.4f}")
    print(f"Symmetric similarity score:        {summary.symmetric_similarity_score:.4f}")
    print(f"Conservative similarity score:     {summary.conservative_similarity_score:.4f}")
    print(f"Pooled similarity score:           {summary.pooled_similarity_score:.4f}")

    print()
    print(f"Median A -> B similarity:          {summary.median_A_to_B_similarity:.4f}")
    print(f"Median B -> A similarity:          {summary.median_B_to_A_similarity:.4f}")
    print(f"P10 A -> B similarity:             {summary.p10_A_to_B_similarity:.4f}")
    print(f"P10 B -> A similarity:             {summary.p10_B_to_A_similarity:.4f}")
    print(f"P90 A -> B similarity:             {summary.p90_A_to_B_similarity:.4f}")
    print(f"P90 B -> A similarity:             {summary.p90_B_to_A_similarity:.4f}")

    print()
    print(f"Fraction A -> B >= 0.90:           {summary.fraction_A_to_B_above_090:.2%}")
    print(f"Fraction B -> A >= 0.90:           {summary.fraction_B_to_A_above_090:.2%}")
    print(f"Fraction A -> B >= 0.95:           {summary.fraction_A_to_B_above_095:.2%}")
    print(f"Fraction B -> A >= 0.95:           {summary.fraction_B_to_A_above_095:.2%}")
    print(f"Fraction A -> B >= 0.99:           {summary.fraction_A_to_B_above_099:.2%}")
    print(f"Fraction B -> A >= 0.99:           {summary.fraction_B_to_A_above_099:.2%}")


def cross_dataset_similarity_summary_to_dict(
    summary: CrossDatasetSimilaritySummary,
) -> dict[str, Any]:
    return {
        "num_A": summary.num_A,
        "num_B": summary.num_B,
        "embedding_dim": summary.embedding_dim,
        "top_k": summary.top_k,
        "chunk_size": summary.chunk_size,

        "A_to_B_score": summary.A_to_B_score,
        "B_to_A_score": summary.B_to_A_score,
        "symmetric_similarity_score": summary.symmetric_similarity_score,
        "conservative_similarity_score": summary.conservative_similarity_score,
        "pooled_similarity_score": summary.pooled_similarity_score,

        "median_A_to_B_similarity": summary.median_A_to_B_similarity,
        "median_B_to_A_similarity": summary.median_B_to_A_similarity,
        "p10_A_to_B_similarity": summary.p10_A_to_B_similarity,
        "p10_B_to_A_similarity": summary.p10_B_to_A_similarity,
        "p90_A_to_B_similarity": summary.p90_A_to_B_similarity,
        "p90_B_to_A_similarity": summary.p90_B_to_A_similarity,

        "min_A_to_B_similarity": summary.min_A_to_B_similarity,
        "min_B_to_A_similarity": summary.min_B_to_A_similarity,
        "max_A_to_B_similarity": summary.max_A_to_B_similarity,
        "max_B_to_A_similarity": summary.max_B_to_A_similarity,

        "fraction_A_to_B_above_090": summary.fraction_A_to_B_above_090,
        "fraction_B_to_A_above_090": summary.fraction_B_to_A_above_090,
        "fraction_A_to_B_above_095": summary.fraction_A_to_B_above_095,
        "fraction_B_to_A_above_095": summary.fraction_B_to_A_above_095,
        "fraction_A_to_B_above_099": summary.fraction_A_to_B_above_099,
        "fraction_B_to_A_above_099": summary.fraction_B_to_A_above_099,
    }