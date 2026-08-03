from typing import Any

import numpy as np
import pytest

from prelims_cli.embedding.inference import (
    LANGUAGE_MODELS,
    OnnxEmbedder,
    _cls_pool,
    _l2_normalize,
    _mean_pool,
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
