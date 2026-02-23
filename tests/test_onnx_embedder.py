import numpy as np

from prelims_cli.embedding.inference import _l2_normalize, _mean_pool


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
