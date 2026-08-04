from typing import Any

import numpy as np
import pytest

from prelims_cli.embedding.inference import (
    LANGUAGE_MODELS,
    OnnxEmbedder,
    _cls_pool,
    _l2_normalize,
    _mean_pool,
    _plan_batches,
)


def test_mean_pool_basic() -> None:
    # (1, 3, 2) hidden states, all tokens active
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 1]], dtype=np.int64)
    result = _mean_pool(hidden, mask)
    expected = np.array([[3.0, 4.0]], dtype=np.float32)  # mean over tokens
    np.testing.assert_array_almost_equal(result, expected)


def test_mean_pool_with_padding() -> None:
    # 2 real tokens, 1 padding token
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    result = _mean_pool(hidden, mask)
    expected = np.array([[2.0, 3.0]], dtype=np.float32)  # mean of first 2 only
    np.testing.assert_array_almost_equal(result, expected)


def test_mean_pool_batch() -> None:
    hidden = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[1, 1], [1, 1]], dtype=np.int64)
    result = _mean_pool(hidden, mask)
    expected = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_cls_pool_takes_first_token() -> None:
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    result = _cls_pool(hidden)
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_cls_pool_batch_ignores_padding() -> None:
    # Padding is right-aligned, so the CLS token is unaffected by it
    hidden = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]],
            [[0.0, 2.0], [0.0, 4.0], [99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    result = _cls_pool(hidden)
    expected = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _FakeTokenizer:
    def __init__(self, encodings: list[_FakeEncoding]) -> None:
        self._encodings = encodings

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        return self._encodings


class _FakeSession:
    def __init__(self, hidden: np.ndarray) -> None:
        self.hidden = hidden

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.hidden]


def _embedder_with(pooling: str, hidden: np.ndarray) -> OnnxEmbedder:
    """Build an embedder around fakes, skipping the model download."""
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder.session = _FakeSession(hidden)  # type: ignore[assignment]
    embedder.tokenizer = _FakeTokenizer(  # type: ignore[assignment]
        [_FakeEncoding([101, 7, 8])]
    )
    embedder.pooling = pooling
    embedder.prefix = ""
    return embedder


HIDDEN = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)


def test_embed_uses_cls_pooling() -> None:
    result = _embedder_with("cls", HIDDEN).embed(["hello"])
    np.testing.assert_array_almost_equal(result, _l2_normalize(_cls_pool(HIDDEN)))


def test_embed_uses_mean_pooling() -> None:
    mask = np.ones((1, 3), dtype=np.int64)
    result = _embedder_with("mean", HIDDEN).embed(["hello"])
    np.testing.assert_array_almost_equal(
        result, _l2_normalize(_mean_pool(HIDDEN, mask))
    )


def test_embed_pooling_methods_differ() -> None:
    """The two methods must not collapse onto the same vector."""
    cls_result = _embedder_with("cls", HIDDEN).embed(["hello"])
    mean_result = _embedder_with("mean", HIDDEN).embed(["hello"])
    assert not np.allclose(cls_result, mean_result)


def test_pooling_is_required() -> None:
    with pytest.raises(TypeError):
        OnnxEmbedder()  # type: ignore[call-arg]


def test_invalid_pooling_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported pooling"):
        OnnxEmbedder(pooling="max")


def test_language_models_declare_pooling() -> None:
    assert LANGUAGE_MODELS["ja"]["pooling"] == "mean"
    assert LANGUAGE_MODELS["en"]["pooling"] == "cls"


def test_plan_batches_covers_every_index_once() -> None:
    lengths = [10, 3, 7, 1, 5, 20, 2]
    batches = _plan_batches(lengths, token_budget=20, max_batch=4)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_plan_batches_respects_token_budget() -> None:
    lengths = [4, 4, 4, 4, 4, 4]
    batches = _plan_batches(lengths, token_budget=8, max_batch=100)
    for batch in batches:
        assert len(batch) * max(lengths[i] for i in batch) <= 8
    assert len(batches) == 3


def test_plan_batches_respects_max_batch() -> None:
    lengths = [1] * 10
    batches = _plan_batches(lengths, token_budget=1000, max_batch=3)
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_plan_batches_keeps_oversized_text_alone() -> None:
    """A text over budget on its own must still be embedded, not dropped."""
    lengths = [2, 2, 500]
    batches = _plan_batches(lengths, token_budget=8, max_batch=100)
    assert [2] in batches
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_plan_batches_groups_similar_lengths() -> None:
    """The point of sorting: padding stays close to the real token count.

    Interleaved short and long texts would otherwise pad every batch up to
    the long one — the failure mode this batching exists to avoid.
    """
    lengths = [1, 100, 1, 100, 1, 100]
    batches = _plan_batches(lengths, token_budget=200, max_batch=3)
    padded = sum(len(b) * max(lengths[i] for i in b) for b in batches)
    assert padded <= sum(lengths) * 1.1


class _LengthTokenizer:
    """Tokenizes each text into ``len(text)`` copies of its own length.

    Lets a test tell embeddings apart by the length of the text they came
    from, which is what checking the restored order needs.
    """

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        return [_FakeEncoding([len(t)] * len(t)) for t in texts]


class _IdentitySession:
    """Returns hidden states carrying each row's first token id."""

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ids = inputs["input_ids"]
        first = ids[:, :1].astype(np.float32)
        hidden = np.repeat(first[:, :, np.newaxis], ids.shape[1], axis=1)
        return [np.concatenate([hidden, np.ones_like(hidden)], axis=2)]


def test_embed_all_returns_input_order() -> None:
    """Batching sorts by length, so the result has to be put back."""
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder.session = _IdentitySession()  # type: ignore[assignment]
    embedder.tokenizer = _LengthTokenizer()  # type: ignore[assignment]
    embedder.pooling = "cls"
    embedder.prefix = ""
    embedder.batch_size = 8
    embedder.token_budget = 8

    texts = ["a" * n for n in (5, 1, 3, 2, 4)]
    result = embedder.embed_all(texts)

    expected = _l2_normalize(
        np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"token_budget": 0}, "token_budget must be at least 1"),
        ({"token_budget": -1}, "token_budget must be at least 1"),
    ],
)
def test_non_positive_batching_limits_raise(kwargs: dict, message: str) -> None:
    """Zero degrades to one text per batch instead of failing, so reject it."""
    with pytest.raises(ValueError, match=message):
        OnnxEmbedder(pooling="mean", **kwargs)


def test_embed_all_rejects_a_batch_of_the_wrong_size() -> None:
    """A short result would otherwise leave Nones and fail inside np.stack."""

    class _ShortSession:
        def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list:
            return [np.ones((1, 2, 2), dtype=np.float32)]

    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder.session = _ShortSession()  # type: ignore[assignment]
    embedder.tokenizer = _LengthTokenizer()  # type: ignore[assignment]
    embedder.pooling = "cls"
    embedder.prefix = ""
    embedder.batch_size = 8
    embedder.token_budget = 4096

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        embedder.embed_all(["aa", "bb"])


def test_embed_all_handles_empty_input() -> None:
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    assert embedder.embed_all([]).shape == (0, 0)


def test_l2_normalize() -> None:
    x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    result = _l2_normalize(x)
    # Check unit norm
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_array_almost_equal(norms, [1.0, 1.0])
    # Check direction preserved
    np.testing.assert_array_almost_equal(result[0], [0.6, 0.8])
    np.testing.assert_array_almost_equal(result[1], [0.0, 1.0])


def test_l2_normalize_zero_vector() -> None:
    x = np.array([[0.0, 0.0]], dtype=np.float32)
    result = _l2_normalize(x)
    # Should not produce NaN
    assert not np.any(np.isnan(result))
