from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Pooling is a per-model property, not a global one: ruri-v3 ships a
# 1_Pooling/config.json that selects mean pooling, while granite-embedding r2
# is trained for CLS pooling (its model card reads model_output[0][:, 0]).
# Using the wrong one does not raise, it just degrades the embedding quality,
# so every entry must state it explicitly and there is no default.
POOLING_METHODS = ("mean", "cls")

# Bump this whenever a change to embed() makes it return different vectors for
# the same input — a new normalization, different truncation, a pooling bug fix.
# It is part of the cache key, so bumping it re-embeds every cached article.
# Model, pooling and prefix are already in that key; this covers the code, which
# no setting reflects. Forget to bump it and sites keep serving stale vectors.
EMBEDDING_CACHE_VERSION = 1

# Revisions are pinned because an unpinned repo resolves to whatever `main` is
# at download time. The cache DB keys on the model name, not on its weights, so
# an upstream re-upload would leave old and new vectors side by side in the same
# database and they would be compared to each other — dimensions match, nothing
# raises, the recommendations are just quietly wrong. The pin is part of the
# cache key, so bumping it re-embeds everything and keeps one generation.
LANGUAGE_MODELS = {
    "ja": {
        "model_name": "sirasagi62/ruri-v3-30m-ONNX",
        "model_file": "onnx/model_quantized.onnx",
        "revision": "cdf9391f1ff2198daa8f63f7ccf97d7b3e7415a0",
        "pooling": "mean",
    },
    "en": {
        "model_name": "onnx-community/granite-embedding-small-english-r2-ONNX",
        "model_file": "onnx/model_quantized.onnx",
        "revision": "1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8",
        "pooling": "cls",
    },
}
DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL_NAME = LANGUAGE_MODELS[DEFAULT_LANGUAGE]["model_name"]
DEFAULT_MODEL_FILE = LANGUAGE_MODELS[DEFAULT_LANGUAGE]["model_file"]
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
MAX_LENGTH = 8192

# Batches are padded to their longest member and attention memory grows with
# the square of that length, so a fixed number of texts per batch makes peak
# memory depend on which texts happen to land together. Article lengths are
# heavily skewed — on a 419-post corpus truncated at 8000 chars the median is
# 545 tokens and the longest is 3636 — so with a fixed batch of 8 in file
# order, nearly every batch caught a long post and got padded up to it: twice
# the token slots and three times the attention work actually needed.
#
# Batching to a budget of padded token slots instead bounds that. Sorting by
# length first is what makes the budget effective: neighbours have similar
# lengths, so little padding is wasted, and long texts end up in small batches
# while short ones pack into large ones. The order is restored afterwards.
TOKEN_BUDGET = 4096


class OnnxEmbedder:
    """Embedding model using ONNX Runtime for CPU inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_file: str = DEFAULT_MODEL_FILE,
        *,
        pooling: str,
        revision: str | None = None,
        prefix: str = "",
        batch_size: int = 8,
        token_budget: int = TOKEN_BUDGET,
    ) -> None:
        if pooling not in POOLING_METHODS:
            raise ValueError(
                f"Unsupported pooling: {pooling!r}. Supported: {list(POOLING_METHODS)}"
            )
        # Both are site-configurable. Zero or negative values do not fail on
        # their own — they quietly degrade to one text per batch — so reject
        # them here, where the setting came from.
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be at least 1, got {token_budget}")

        import onnxruntime as ort  # type: ignore[import-not-found,import-untyped]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        from huggingface_hub.utils import (  # type: ignore[import-not-found]
            EntryNotFoundError,
        )
        from tokenizers import (  # type: ignore[import-not-found,import-untyped]
            Tokenizer,
        )

        model_path = hf_hub_download(
            repo_id=model_name, filename=model_file, revision=revision
        )
        tokenizer_path = hf_hub_download(
            repo_id=model_name, filename=DEFAULT_TOKENIZER_FILE, revision=revision
        )

        # Some ONNX models store weights in a companion _data file
        try:
            hf_hub_download(
                repo_id=model_name, filename=f"{model_file}_data", revision=revision
            )
        except EntryNotFoundError:
            pass

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.pooling = pooling
        self.revision = revision
        self.prefix = prefix
        self.batch_size = batch_size
        self.token_budget = token_budget

    def embed_all(self, texts: list[str]) -> np.ndarray:
        """Embed texts in memory-bounded batches, in input order.

        Prefer this over calling embed() in a loop: it groups texts of similar
        length together so batches are not padded up to an unrelated long one,
        and caps each batch by padded token slots rather than by count.

        Returns an (N, dim) float32 array.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        lengths = self._token_lengths(texts)
        embeddings: list[np.ndarray | None] = [None] * len(texts)
        for batch in _plan_batches(lengths, self.token_budget, self.batch_size):
            # strict=True so a batch that comes back the wrong size fails here,
            # naming the cause, rather than later as a stack of Nones.
            rows = self.embed([texts[i] for i in batch])
            for i, embedding in zip(batch, rows, strict=True):
                embeddings[i] = embedding

        filled = [e for e in embeddings if e is not None]
        if len(filled) != len(texts):
            raise RuntimeError(
                f"batching covered {len(filled)} of {len(texts)} texts; "
                "every text must land in exactly one batch"
            )
        return np.stack(filled)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute L2-normalized embeddings for a list of texts as one batch.

        Every text is padded to the longest one here, so pass a batch that
        embed_all() planned rather than an arbitrary list.

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

        if self.pooling == "cls":
            embeddings = _cls_pool(last_hidden_state)
        else:
            embeddings = _mean_pool(last_hidden_state, attention_mask)
        embeddings = _l2_normalize(embeddings)
        return embeddings

    def _token_lengths(self, texts: list[str]) -> list[int]:
        """Token count per text, as embed() would see it.

        The prefix is included because it is prepended before tokenizing, and
        truncation is already enabled, so these are capped at MAX_LENGTH.
        """
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        return [len(e.ids) for e in self.tokenizer.encode_batch(texts)]

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


def _plan_batches(
    lengths: list[int], token_budget: int, max_batch: int
) -> list[list[int]]:
    """Group indices into batches of similar length under a token budget.

    A batch costs ``len(batch) * max(lengths in batch)`` token slots once
    padded; batches are grown until adding the next text would exceed
    ``token_budget`` slots or ``max_batch`` texts. Sorting by length first
    keeps that padding close to the real token count.

    A text longer than the budget on its own gets a batch to itself rather
    than being dropped or split — truncation to MAX_LENGTH is the only thing
    that bounds it.

    Args:
        lengths: token count per text, indexed as the caller's list
        token_budget: maximum padded token slots per batch
        max_batch: maximum texts per batch

    Returns:
        Lists of indices into ``lengths``, together covering every index once.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        padded_max = max(current_max, lengths[i])
        if current and (
            len(current) + 1 > max_batch
            or (len(current) + 1) * padded_max > token_budget
        ):
            batches.append(current)
            current, padded_max = [], lengths[i]
        current.append(i)
        current_max = padded_max
    if current:
        batches.append(current)
    return batches


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


def _cls_pool(last_hidden_state: np.ndarray) -> np.ndarray:
    """Take the first ([CLS]) token embedding of each sequence.

    Padding is right-aligned, so index 0 is always a real token.

    Args:
        last_hidden_state: (N, seq_len, dim) float array

    Returns:
        (N, dim) float32 array
    """
    return last_hidden_state[:, 0].astype(np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row of x."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms
