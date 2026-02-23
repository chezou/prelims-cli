from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

LANGUAGE_MODELS = {
    "ja": {
        "model_name": "sirasagi62/ruri-v3-30m-ONNX",
        "model_file": "onnx/model_quantized.onnx",
    },
    "en": {
        "model_name": "onnx-community/granite-embedding-small-english-r2-ONNX",
        "model_file": "onnx/model_quantized.onnx",
    },
}
DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL_NAME = LANGUAGE_MODELS[DEFAULT_LANGUAGE]["model_name"]
DEFAULT_MODEL_FILE = LANGUAGE_MODELS[DEFAULT_LANGUAGE]["model_file"]
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
MAX_LENGTH = 8192


class OnnxEmbedder:
    """Embedding model using ONNX Runtime for CPU inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_file: str = DEFAULT_MODEL_FILE,
        prefix: str = "",
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found,import-untyped]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from huggingface_hub.utils import (  # type: ignore[import-not-found]
            EntryNotFoundError,
        )
        from tokenizers import (  # type: ignore[import-not-found,import-untyped]
            Tokenizer,
        )

        model_path = hf_hub_download(repo_id=model_name, filename=model_file)
        tokenizer_path = hf_hub_download(
            repo_id=model_name, filename=DEFAULT_TOKENIZER_FILE
        )

        # Some ONNX models store weights in a companion _data file
        try:
            hf_hub_download(repo_id=model_name, filename=f"{model_file}_data")
        except EntryNotFoundError:
            pass

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.prefix = prefix

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute L2-normalized embeddings for a list of texts.

        Returns an (N, dim) float32 array.
        """
        if self.prefix:
            texts = [self.prefix + t for t in texts]

        input_ids, attention_mask = self._tokenize(texts)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        last_hidden_state = outputs[0]  # (N, seq_len, dim)

        embeddings = _mean_pool(last_hidden_state, attention_mask)
        embeddings = _l2_normalize(embeddings)
        return embeddings

    def _tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize texts and return input_ids and attention_mask arrays.

        Padding is done per-batch to the longest sequence in the batch,
        rather than globally to MAX_LENGTH, to reduce peak memory usage.
        """
        encodings = self.tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, e in enumerate(encodings):
            length = len(e.ids)
            input_ids[i, :length] = e.ids
            attention_mask[i, :length] = e.attention_mask
        return input_ids, attention_mask


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling over token embeddings, respecting attention mask.

    Args:
        last_hidden_state: (N, seq_len, dim) float array
        attention_mask: (N, seq_len) int array

    Returns:
        (N, dim) float32 array
    """
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return (summed / counts).astype(np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row of x."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms
