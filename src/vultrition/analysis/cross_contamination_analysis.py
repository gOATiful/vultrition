import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def embedding_dataset_similarity(
    A: np.ndarray,
    B: np.ndarray,
    ids_A=None,
    ids_B=None,
    top_k: int = 1,
):
    """
    Compute a sample-level similarity score between two embedding datasets.

    Main idea:
        For each sample in A, find its nearest neighbor in B.
        For each sample in B, find its nearest neighbor in A.
        Average both directions.

    Returns:
        summary: dict with dataset-level scores
        A_to_B_matches: DataFrame with best B matches for A samples
        B_to_A_matches: DataFrame with best A matches for B samples
    """

    A = normalize(np.asarray(A))
    B = normalize(np.asarray(B))

    n_A = len(A)
    n_B = len(B)

    if ids_A is None:
        ids_A = np.arange(n_A)

    if ids_B is None:
        ids_B = np.arange(n_B)

    ids_A = np.asarray(ids_A)
    ids_B = np.asarray(ids_B)

    # A -> B
    nn_B = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn_B.fit(B)

    dist_A_to_B, idx_A_to_B = nn_B.kneighbors(A)
    sim_A_to_B = 1.0 - dist_A_to_B

    # B -> A
    nn_A = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn_A.fit(A)

    dist_B_to_A, idx_B_to_A = nn_A.kneighbors(B)
    sim_B_to_A = 1.0 - dist_B_to_A

    # If top_k = 1, this is just nearest-neighbor similarity.
    # If top_k > 1, this averages the top-k nearest similarities per sample.
    per_sample_A_to_B = sim_A_to_B.mean(axis=1)
    per_sample_B_to_A = sim_B_to_A.mean(axis=1)

    A_to_B_score = float(per_sample_A_to_B.mean())
    B_to_A_score = float(per_sample_B_to_A.mean())

    symmetric_similarity_score = float(
        0.5 * (A_to_B_score + B_to_A_score)
    )

    conservative_similarity_score = float(
        min(A_to_B_score, B_to_A_score)
    )

    summary = {
        "A_to_B_score": A_to_B_score,
        "B_to_A_score": B_to_A_score,
        "symmetric_similarity_score": symmetric_similarity_score,
        "conservative_similarity_score": conservative_similarity_score,
        "median_A_to_B_similarity": float(np.median(per_sample_A_to_B)),
        "median_B_to_A_similarity": float(np.median(per_sample_B_to_A)),
        "p10_A_to_B_similarity": float(np.quantile(per_sample_A_to_B, 0.10)),
        "p10_B_to_A_similarity": float(np.quantile(per_sample_B_to_A, 0.10)),
        "p90_A_to_B_similarity": float(np.quantile(per_sample_A_to_B, 0.90)),
        "p90_B_to_A_similarity": float(np.quantile(per_sample_B_to_A, 0.90)),
        "num_A": n_A,
        "num_B": n_B,
        "top_k": top_k,
    }

    # A_to_B_matches = pd.DataFrame({
    #     "a_index": np.arange(n_A),
    #     "a_id": ids_A,
    #     "best_b_index": idx_A_to_B[:, 0],
    #     "best_b_id": ids_B[idx_A_to_B[:, 0]],
    #     "best_cosine_similarity": sim_A_to_B[:, 0],
    #     "mean_top_k_similarity": per_sample_A_to_B,
    # }).sort_values("best_cosine_similarity", ascending=False)

    # B_to_A_matches = pd.DataFrame({
    #     "b_index": np.arange(n_B),
    #     "b_id": ids_B,
    #     "best_a_index": idx_B_to_A[:, 0],
    #     "best_a_id": ids_A[idx_B_to_A[:, 0]],
    #     "best_cosine_similarity": sim_B_to_A[:, 0],
    #     "mean_top_k_similarity": per_sample_B_to_A,
    # }).sort_values("best_cosine_similarity", ascending=False)

    return summary# , A_to_B_matches, B_to_A_matches