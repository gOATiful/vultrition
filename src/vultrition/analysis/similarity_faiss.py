from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Any, Tuple

import numpy as np


def _top_n_average_metrics(
    similarity_scores: np.ndarray,
    n: int,
    threshold: float,
) -> Dict[str, float]:
    """
    Computes:
    - per-function average of top-n similarities
    - dataset average of those per-function top-n averages
    - ratio/percentage of functions whose top-n average is >= threshold
    """

    if similarity_scores.shape[1] < n:
        return {
            f"average_top{n}_nearest_neighbor_similarity": float("nan"),
            f"top{n}_average_above_threshold_ratio": float("nan"),
            f"top{n}_average_above_threshold_percentage": float("nan"),
        }

    top_n_scores = similarity_scores[:, :n]
    top_n_average_scores = np.nanmean(top_n_scores, axis=1)

    valid_mask = np.isfinite(top_n_average_scores)

    if not np.any(valid_mask):
        return {
            f"average_top{n}_nearest_neighbor_similarity": float("nan"),
            f"top{n}_average_above_threshold_ratio": float("nan"),
            f"top{n}_average_above_threshold_percentage": float("nan"),
        }

    valid_top_n_average_scores = top_n_average_scores[valid_mask]

    average_top_n_similarity = float(np.mean(valid_top_n_average_scores))

    top_n_above_threshold_ratio = float(
        np.mean(valid_top_n_average_scores >= threshold)
    )

    top_n_above_threshold_percentage = float(
        100.0 * top_n_above_threshold_ratio
    )

    return {
        f"average_top{n}_nearest_neighbor_similarity": average_top_n_similarity,
        f"top{n}_average_above_threshold_ratio": top_n_above_threshold_ratio,
        f"top{n}_average_above_threshold_percentage": top_n_above_threshold_percentage,
    }


def assess_function_similarity_dataset(
    embeddings: np.ndarray,
    ids: Optional[Sequence[Union[str, int]]] = None,
    k: int = 20,
    use_gpu: bool = False,
    gpu_id: int = 0,
    batch_size: int = 4096,
    output_csv: Optional[Union[str, Path]] = None,
    duplicate_threshold: float = 0.98,
    near_duplicate_threshold: float = 0.95,
    highly_similar_threshold: float = 0.90,
    nearest_neighbor_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute top-k cosine similarities between function embeddings and return
    dataset-level quality metrics.

    Parameters
    ----------
    embeddings:
        NumPy array with shape (num_functions, embedding_dim).

    ids:
        Optional function IDs. If provided, used in the output CSV.

    k:
        Number of nearest neighbors to keep per function, excluding itself.

    use_gpu:
        If True, use FAISS GPU search.

    gpu_id:
        GPU device ID to use.

    batch_size:
        Number of query vectors searched per batch.

    output_csv:
        Optional CSV path. If provided, writes:
        src_id, dst_id, score, rank

    duplicate_threshold:
        Top-1 similarity >= this value is counted as a likely duplicate.

    near_duplicate_threshold:
        Top-1 similarity >= this value is counted as a likely near-duplicate.

    highly_similar_threshold:
        Top-1 similarity >= this value is counted as highly similar.

    Returns
    -------
    result:
        Dictionary containing:
        - neighbor_indices
        - similarity_scores
        - dataset_quality_score
        - mean_top1_similarity
        - median_top1_similarity
        - duplicate_ratio
        - near_duplicate_ratio
        - highly_similar_ratio
        - top1_percentiles
    """

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required. Install CPU version with `pip install faiss-cpu`, "
            "or install a GPU-enabled FAISS build for GPU support."
        ) from exc

    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a NumPy array")

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape (num_functions, embedding_dim)")

    num_functions, dim = embeddings.shape

    if num_functions < 2:
        raise ValueError("Need at least two embeddings")

    if ids is not None and len(ids) != num_functions:
        raise ValueError("ids must have the same length as embeddings")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # We need at least 5 neighbors to compute top-3 and top-5 metrics.
    k = min(max(k, 3), num_functions - 1)

    vectors = np.ascontiguousarray(embeddings.astype(np.float32, copy=True))

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            "Cosine similarity cannot be computed for zero-vector embeddings")

    # Normalize vectors so inner product == cosine similarity.
    faiss.normalize_L2(vectors)

    cpu_index = faiss.IndexFlatIP(dim)

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "This FAISS installation does not have GPU support. "
                "You likely installed faiss-cpu. Install a GPU-enabled FAISS build."
            )

        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, cpu_index)
    else:
        index = cpu_index

    index.add(vectors)

    neighbor_indices = np.full((num_functions, k), -1, dtype=np.int64)
    similarity_scores = np.full((num_functions, k), np.nan, dtype=np.float32)

    search_k = min(num_functions, k + 1)

    csv_file = None
    writer = None

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        csv_file = output_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["src_id", "dst_id", "score", "rank"])

    try:
        for start in range(0, num_functions, batch_size):
            end = min(start + batch_size, num_functions)
            query_vectors = vectors[start:end]

            scores_batch, indices_batch = index.search(query_vectors, search_k)

            for row_offset, src_index in enumerate(range(start, end)):
                kept = 0

                for score, dst_index in zip(
                    scores_batch[row_offset],
                    indices_batch[row_offset],
                ):
                    dst_index = int(dst_index)

                    # Remove self-match.
                    if dst_index == src_index:
                        continue

                    if dst_index < 0:
                        continue

                    neighbor_indices[src_index, kept] = dst_index
                    similarity_scores[src_index, kept] = float(score)

                    if writer is not None:
                        src_id = ids[src_index] if ids is not None else src_index
                        dst_id = ids[dst_index] if ids is not None else dst_index
                        writer.writerow(
                            [src_id, dst_id, float(score), kept + 1])

                    kept += 1

                    if kept == k:
                        break

    finally:
        if csv_file is not None:
            csv_file.close()

    top1_scores = similarity_scores[:, 0]

    # Score 1:
    # Average TOP-1 nearest-neighbor similarity across all functions.
    average_top1_nearest_neighbor_similarity = float(np.nanmean(top1_scores))

    # Score 2:
    # Percentage of functions whose nearest-neighbor similarity is above
    # the configurable threshold. Default threshold: 0.95.
    nearest_neighbor_above_threshold_ratio = float(
        np.mean(top1_scores >= nearest_neighbor_threshold)
    )

    nearest_neighbor_above_threshold_percentage = float(
        100.0 * nearest_neighbor_above_threshold_ratio
    )

    # New TOP-3 / TOP-5 metrics.
    top3_metrics = _top_n_average_metrics(
        similarity_scores=similarity_scores,
        n=3,
        threshold=nearest_neighbor_threshold,
    )


    mean_top1 = average_top1_nearest_neighbor_similarity
    median_top1 = float(np.nanmedian(top1_scores))

    duplicate_ratio = float(np.mean(top1_scores >= duplicate_threshold))
    near_duplicate_ratio = float(
        np.mean(top1_scores >= near_duplicate_threshold))
    highly_similar_ratio = float(
        np.mean(top1_scores >= highly_similar_threshold))

    top1_percentiles = {
        "p01": float(np.percentile(top1_scores, 1)),
        "p05": float(np.percentile(top1_scores, 5)),
        "p10": float(np.percentile(top1_scores, 10)),
        "p25": float(np.percentile(top1_scores, 25)),
        "p50": float(np.percentile(top1_scores, 50)),
        "p75": float(np.percentile(top1_scores, 75)),
        "p90": float(np.percentile(top1_scores, 90)),
        "p95": float(np.percentile(top1_scores, 95)),
        "p99": float(np.percentile(top1_scores, 99)),
    }

    # Score 1:
    # Average TOP-1 nearest-neighbor similarity across all functions.
    average_top1_nearest_neighbor_similarity = float(np.nanmean(top1_scores))

    # Score 2:
    # Percentage of functions whose nearest-neighbor similarity is above
    # the configurable threshold. Default threshold: 0.95.
    nearest_neighbor_above_threshold_ratio = float(
        np.mean(top1_scores >= nearest_neighbor_threshold)
    )

    nearest_neighbor_above_threshold_percentage = float(
        100.0 * nearest_neighbor_above_threshold_ratio
    )

    mean_top1 = average_top1_nearest_neighbor_similarity
    median_top1 = float(np.nanmedian(top1_scores))

    duplicate_ratio = float(np.mean(top1_scores >= duplicate_threshold))
    near_duplicate_ratio = float(
        np.mean(top1_scores >= near_duplicate_threshold))
    highly_similar_ratio = float(
        np.mean(top1_scores >= highly_similar_threshold))

    top1_percentiles = {
        "p01": float(np.percentile(top1_scores, 1)),
        "p05": float(np.percentile(top1_scores, 5)),
        "p10": float(np.percentile(top1_scores, 10)),
        "p25": float(np.percentile(top1_scores, 25)),
        "p50": float(np.percentile(top1_scores, 50)),
        "p75": float(np.percentile(top1_scores, 75)),
        "p90": float(np.percentile(top1_scores, 90)),
        "p95": float(np.percentile(top1_scores, 95)),
        "p99": float(np.percentile(top1_scores, 99)),
    }

    result = {
        "neighbor_indices": neighbor_indices,
        "similarity_scores": similarity_scores,

        # Requested top-1 scores
        "average_top1_nearest_neighbor_similarity": average_top1_nearest_neighbor_similarity,
        "nearest_neighbor_threshold": nearest_neighbor_threshold,
        "nearest_neighbor_above_threshold_ratio": nearest_neighbor_above_threshold_ratio,
        "nearest_neighbor_above_threshold_percentage": nearest_neighbor_above_threshold_percentage,

        # New top-3 scores
        **top3_metrics,

        # Existing metrics
        "mean_top1_similarity": mean_top1,
        "median_top1_similarity": median_top1,

        "duplicate_ratio": duplicate_ratio,
        "near_duplicate_ratio": near_duplicate_ratio,
        "highly_similar_ratio": highly_similar_ratio,

        "duplicate_threshold": duplicate_threshold,
        "near_duplicate_threshold": near_duplicate_threshold,
        "highly_similar_threshold": highly_similar_threshold,

        "top1_percentiles": top1_percentiles,

        "num_functions": num_functions,
        "embedding_dim": dim,
        "k": k,
        "used_gpu": use_gpu,
    }

    return result
