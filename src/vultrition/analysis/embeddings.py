import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from vultrition.models.dataset import Sample


MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"


@dataclass
class CodeEmbeddingResult:
    embeddings: np.ndarray
    ids: np.ndarray


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def default_sample_id(sample: Sample, index: int) -> str:
    """
    Create a stable-ish readable ID for a sample.

    The index is included so duplicate functions still get distinct IDs.
    The function hash is included so the ID still identifies the code content.
    """

    function_hash = hashlib.sha256(
        sample.function.encode("utf-8")
    ).hexdigest()[:16]

    cwe = ",".join(sample.cwe) if sample.cwe else "no-cwe"
    cve = sample.cve if sample.cve else "no-cve"
    project = sample.project if sample.project else "no-project"

    return (
        f"idx={index}"
        f"|project={project}"
        f"|cve={cve}"
        f"|cwe={cwe}"
        f"|label={sample.label}"
        f"|sha256={function_hash}"
    )


def create_sample_ids(
    samples: Sequence[Sample],
    id_fn: Callable[[Sample, int], str] | None = None,
) -> np.ndarray:
    """
    Create one ID per sample.

    Args:
        samples:
            Original samples.

        id_fn:
            Optional custom function:
                id_fn(sample, index) -> str

            If None, default_sample_id is used.

    Returns:
        NumPy array of shape (num_samples,)
    """

    if id_fn is None:
        id_fn = default_sample_id

    return np.asarray(
        [id_fn(sample, index) for index, sample in enumerate(samples)],
        dtype=object,
    )

@torch.inference_mode()
def create_code_embeddings(
    samples: Sequence[Sample],
    batch_size: int = 8,
    max_length: int = 8192,
    id_fn: Callable[[Sample, int], str] | None = None,
    normalize: bool = False,
) -> CodeEmbeddingResult:
    """
    Create code embeddings and matching sample IDs.

    Args:
        samples:
            Original samples.

        batch_size:
            Number of samples per embedding batch.

        max_length:
            Maximum token length for tokenizer truncation.

        id_fn:
            Optional custom ID function.

        normalize:
            If True, L2-normalize embeddings after all embeddings are created.

    Returns:
        CodeEmbeddingResult:
            embeddings:
                NumPy array with shape (num_samples, embedding_dim)

            ids:
                NumPy array with shape (num_samples,)

    Important:
        embeddings[i] corresponds to ids[i] and samples[i].
    """

    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)

    device = get_device()
    dtype = get_dtype(device)

    sample_ids = create_sample_ids(samples, id_fn=id_fn)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"CPU threads: {cpu_count}")
    print(f"Samples: {len(samples)}")
    print(f"Batch size: {batch_size}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    print("Creating embeddings...")

    all_embeddings: list[np.ndarray] = []
    prefix = "Candidate code snippet:\n"

    for start in tqdm(
        range(0, len(samples), batch_size),
        total=(len(samples) + batch_size - 1) // batch_size,
        desc="Embedding batches",
        unit="batch",
    ):
        batch = samples[start : start + batch_size]
        texts = [prefix + sample.function for sample in batch]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encoded)

        embeddings = last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        all_embeddings.append(
            embeddings.float().cpu().numpy().astype(np.float32)
        )

    embeddings_np = np.vstack(all_embeddings)

    if normalize:
        norms = np.linalg.norm(embeddings_np, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings_np = embeddings_np / norms

    if len(embeddings_np) != len(sample_ids):
        raise RuntimeError(
            f"Embedding/id mismatch: {len(embeddings_np)} embeddings, "
            f"{len(sample_ids)} ids"
        )

    print("Done.")
    print(f"Embedding shape: {embeddings_np.shape}")
    print(f"IDs shape: {sample_ids.shape}")

    return CodeEmbeddingResult(
        embeddings=embeddings_np.astype(np.float32),
        ids=sample_ids,
    )