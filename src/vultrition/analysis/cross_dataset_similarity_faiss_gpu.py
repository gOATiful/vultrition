from __future__ import annotations

"""
FAISS GPU implementation of cross-dataset embedding similarity.

This module is intended as a drop-in replacement for a NumPy/chunked
implementation that computes bidirectional cross-dataset cosine similarity.

Main behavior:
    - validates two embedding matrices A and B
    - optionally L2-normalizes them on CPU with NumPy
    - builds a FAISS IndexFlatIP index over B and queries A
    - builds a FAISS IndexFlatIP index over A and queries B
    - reports directional, symmetric, conservative, and pooled similarity scores

Cosine similarity note:
    FAISS IndexFlatIP performs maximum inner-product search. If embeddings are
    L2-normalized first, inner product is equivalent to cosine similarity.

GPU note:
    This file uses FAISS GPU by default. Your installed FAISS build must be
    compatible with the CUDA version and GPU compute capability on the node.
    If you see CUDA error 209, the FAISS binary was likely not built for that
    GPU architecture. That cannot be fixed in Python code; use a compatible
    FAISS/CUDA build or request a compatible GPU node.
"""

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

    return np.ascontiguousarray(X.astype(np.float32, copy=False))


def _l2_normalize_numpy(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows using NumPy on CPU.

    This intentionally avoids faiss.normalize_L2 because some FAISS GPU builds
    may route normalization through CUDA kernels. Normalizing on CPU makes this
    step independent of FAISS GPU compatibility.
    """

    X = np.ascontiguousarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return np.ascontiguousarray(X / norms, dtype=np.float32)


def _import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FAISS is required. Install a GPU-enabled FAISS build for GPU use, "
            "or install faiss-cpu for CPU use."
        ) from exc

    return faiss


def _faiss_has_gpu(faiss: Any) -> bool:
    return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "get_num_gpus")


def _describe_faiss_environment(faiss: Any) -> str:
    parts: list[str] = []

    version = getattr(faiss, "__version__", None)
    if version is not None:
        parts.append(f"faiss_version={version}")

    if hasattr(faiss, "get_num_gpus"):
        try:
            parts.append(f"num_gpus={faiss.get_num_gpus()}")
        except Exception:
            parts.append("num_gpus=unknown")
    else:
        parts.append("num_gpus=unavailable")

    parts.append(f"gpu_api_available={_faiss_has_gpu(faiss)}")
    return ", ".join(parts)


def _build_faiss_index_flat_ip(
    target: np.ndarray,
    *,
    use_gpu: bool,
    gpu_id: int,
    gpu_float16: bool,
    gpu_temp_memory_mb: int | None,
):
    """
    Build a FAISS IndexFlatIP index.

    Returns:
        index:
            FAISS index, CPU or GPU.

        gpu_resources:
            FAISS StandardGpuResources object, or None on CPU.
            Keep this object alive while the GPU index is alive.
    """

    faiss = _import_faiss()

    target = np.ascontiguousarray(target, dtype=np.float32)
    dim = target.shape[1]

    if not use_gpu:
        index = faiss.IndexFlatIP(dim)
        index.add(target)
        return index, None

    if not _faiss_has_gpu(faiss):
        raise RuntimeError(
            "This FAISS installation does not expose GPU APIs. "
            f"FAISS environment: {_describe_faiss_environment(faiss)}"
        )

    num_gpus = faiss.get_num_gpus()
    if num_gpus <= 0:
        raise RuntimeError(
            "FAISS GPU support is present, but no GPUs are visible. "
            "Check your SLURM --gres setting, CUDA_VISIBLE_DEVICES, and nvidia-smi. "
            f"FAISS environment: {_describe_faiss_environment(faiss)}"
        )

    if gpu_id < 0 or gpu_id >= num_gpus:
        raise ValueError(
            f"gpu_id={gpu_id} is invalid; FAISS sees {num_gpus} GPU(s)."
        )

    gpu_resources = faiss.StandardGpuResources()

    if gpu_temp_memory_mb is not None:
        if gpu_temp_memory_mb < 0:
            raise ValueError("gpu_temp_memory_mb must be non-negative or None")
        gpu_resources.setTempMemory(int(gpu_temp_memory_mb) * 1024 * 1024)

    # Prefer explicit GPU index construction instead of index_cpu_to_gpu because
    # it gives us direct control over float16 storage.
    config = faiss.GpuIndexFlatConfig()
    config.device = gpu_id
    config.useFloat16 = bool(gpu_float16)

    try:
        index = faiss.GpuIndexFlatIP(gpu_resources, dim, config)
        index.add(target)
    except RuntimeError as exc:
        msg = str(exc)
        if "no kernel image is available" in msg or "CUDA error 209" in msg:
            raise RuntimeError(
                "FAISS GPU failed with CUDA error 209: no kernel image is available "
                "for execution on the device. This means the installed FAISS/CUDA "
                "binary is not compatible with the GPU architecture on this node. "
                "Request a compatible GPU node or reinstall/build FAISS for this GPU. "
                "This cannot be fixed by changing the similarity code alone. "
                f"Original error: {msg}"
            ) from exc
        raise

    return index, gpu_resources


def _directional_similarity_faiss(
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
    use_gpu: bool,
    gpu_id: int,
    gpu_float16: bool,
    gpu_temp_memory_mb: int | None,
) -> DirectionalSimilarityResult:
    """
    For every row in source, find top-k nearest rows in target using FAISS.

    Assumes source and target are already L2-normalized if cosine similarity
    is desired. With normalized vectors, inner product == cosine similarity.
    """

    num_source = source.shape[0]
    num_target = target.shape[0]

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    effective_top_k = min(top_k, num_target)

    source = np.ascontiguousarray(source, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)

    index, gpu_resources = _build_faiss_index_flat_ip(
        target,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        gpu_float16=gpu_float16,
        gpu_temp_memory_mb=gpu_temp_memory_mb,
    )

    # Keep gpu_resources referenced until after the search loop.
    _ = gpu_resources

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

    try:
        iterator = range(0, num_source, chunk_size)
        for start in tqdm(
            iterator,
            total=(num_source + chunk_size - 1) // chunk_size,
            desc=desc,
            unit="chunk",
            disable=not show_progress,
        ):
            end = min(start + chunk_size, num_source)
            source_chunk = np.ascontiguousarray(source[start:end], dtype=np.float32)

            try:
                topk_values, topk_indices = index.search(source_chunk, effective_top_k)
            except RuntimeError as exc:
                msg = str(exc)
                if "no kernel image is available" in msg or "CUDA error 209" in msg:
                    raise RuntimeError(
                        "FAISS GPU search failed with CUDA error 209: no kernel image "
                        "is available for execution on the device. The installed FAISS "
                        "GPU binary is incompatible with this GPU architecture. "
                        "Request a compatible GPU node or install/build FAISS for this GPU. "
                        f"Original error: {msg}"
                    ) from exc
                raise

            best_target_indices[start:end] = topk_indices[:, 0].astype(np.int64)
            best_cosine_similarity[start:end] = topk_values[:, 0].astype(np.float32)
            mean_top_k_similarity[start:end] = topk_values.mean(axis=1).astype(np.float32)

            if return_topk:
                all_topk_indices[start:end] = topk_indices.astype(np.int64)
                all_topk_sims[start:end] = topk_values.astype(np.float32)

    finally:
        # Explicitly release references before building the opposite-direction index.
        del index

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

    symmetric_similarity_score = float(0.5 * (A_to_B_score + B_to_A_score))
    conservative_similarity_score = float(min(A_to_B_score, B_to_A_score))
    pooled_similarity_score = float(
        (per_sample_A_to_B.sum() + per_sample_B_to_A.sum()) / (num_A + num_B)
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
    chunk_size: int = 8192,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
    return_matches: bool = False,
    return_topk: bool = False,
    use_gpu: bool = True,
    gpu_id: int = 0,
    gpu_float16: bool = False,
    gpu_temp_memory_mb: int | None = None,
) -> CrossDatasetSimilarityResult:
    """
    Compute exact cross-dataset cosine similarity using FAISS.

    Main idea:
        For each sample in A, find its top-k nearest samples in B.
        For each sample in B, find its top-k nearest samples in A.
        Average both directions.

    This version uses FAISS IndexFlatIP. With normalize_embeddings=True,
    inner product equals cosine similarity.

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
            Number of source rows queried at once. Larger is faster but uses
            more GPU memory. Try 4096, 8192, or 16384.

        normalize_embeddings:
            If True, L2-normalize embeddings first on CPU using NumPy.

        show_progress:
            Show tqdm progress bars.

        return_matches:
            If True, return per-sample best matches.

        return_topk:
            If True, store all top-k matches. This can use extra memory.

        use_gpu:
            If True, use FAISS GPU. Defaults to True.

        gpu_id:
            GPU device ID visible to FAISS. Usually 0 inside a SLURM job where
            CUDA_VISIBLE_DEVICES exposes one GPU.

        gpu_float16:
            If True, store vectors in float16 on the GPU index. This can reduce
            GPU memory usage but may slightly change similarity scores.

        gpu_temp_memory_mb:
            Optional FAISS temporary memory limit in MB. Leave None unless you
            need to control GPU memory usage.

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

    if ids_A is not None:
        ids_A = np.asarray(ids_A).reshape(-1)
        if ids_A.shape[0] != num_A:
            raise ValueError(f"ids_A has length {ids_A.shape[0]}, expected {num_A}")

    if ids_B is not None:
        ids_B = np.asarray(ids_B).reshape(-1)
        if ids_B.shape[0] != num_B:
            raise ValueError(f"ids_B has length {ids_B.shape[0]}, expected {num_B}")

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    if normalize_embeddings:
        print("Normalizing embeddings on CPU...")
        A = _l2_normalize_numpy(A)
        B = _l2_normalize_numpy(B)

    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    faiss = _import_faiss()
    backend = "faiss/gpu" if use_gpu else "faiss/cpu"

    print(f"A shape:      {A.shape}")
    print(f"B shape:      {B.shape}")
    print(f"top_k:        {top_k}")
    print(f"chunk_size:   {chunk_size}")
    print(f"backend:      {backend}")
    print(f"FAISS:        {_describe_faiss_environment(faiss)}")
    if use_gpu:
        print(f"gpu_id:       {gpu_id}")
        print(f"gpu_float16:  {gpu_float16}")
        if gpu_temp_memory_mb is not None:
            print(f"gpu_temp_mem: {gpu_temp_memory_mb} MB")

    A_to_B = _directional_similarity_faiss(
        source=A,
        target=B,
        source_ids=ids_A,
        target_ids=ids_B,
        top_k=top_k,
        chunk_size=chunk_size,
        desc="A -> B similarity",
        show_progress=show_progress,
        return_topk=return_topk,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        gpu_float16=gpu_float16,
        gpu_temp_memory_mb=gpu_temp_memory_mb,
    )

    B_to_A = _directional_similarity_faiss(
        source=B,
        target=A,
        source_ids=ids_B,
        target_ids=ids_A,
        top_k=top_k,
        chunk_size=chunk_size,
        desc="B -> A similarity",
        show_progress=show_progress,
        return_topk=return_topk,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        gpu_float16=gpu_float16,
        gpu_temp_memory_mb=gpu_temp_memory_mb,
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


def print_faiss_gpu_debug() -> None:
    """
    Small helper to print FAISS GPU visibility inside a SLURM job.
    """

    faiss = _import_faiss()
    print(f"FAISS: {_describe_faiss_environment(faiss)}")

    if hasattr(faiss, "get_num_gpus"):
        try:
            print(f"FAISS visible GPUs: {faiss.get_num_gpus()}")
        except Exception as exc:
            print(f"Could not query FAISS GPUs: {exc}")


__all__ = [
    "DirectionalSimilarityResult",
    "CrossDatasetSimilaritySummary",
    "CrossDatasetSimilarityResult",
    "embedding_dataset_similarity",
    "print_cross_dataset_similarity_summary",
    "cross_dataset_similarity_summary_to_dict",
    "print_faiss_gpu_debug",
]
