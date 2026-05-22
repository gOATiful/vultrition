import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


from vultrition.models.dataset import Sample


MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU backend
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


@torch.inference_mode()
def create_code_embeddings(
    samples: Sequence[Sample],
    batch_size: int = 8,
    max_length: int = 8192,
) -> np.ndarray:
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)

    device = get_device()
    dtype = get_dtype(device)

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

        embeddings = F.normalize(embeddings.float(), p=2, dim=1)

        all_embeddings.append(
            embeddings.cpu().numpy().astype(np.float32)
        )

    result = np.vstack(all_embeddings)

    print("Done.")
    print(f"Embedding shape: {result.shape}")

    return result