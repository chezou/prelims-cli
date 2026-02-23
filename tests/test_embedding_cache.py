from pathlib import Path

import numpy as np

from prelims_cli.embedding.cache import EmbeddingCache


def _make_cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(str(tmp_path / "test.db"))


def test_put_and_get(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    emb = np.random.rand(256).astype(np.float32)
    cache.put("a.md", "hash1", emb)
    result = cache.get("a.md", "hash1")
    assert result is not None
    np.testing.assert_array_almost_equal(result, emb)
    cache.close()


def test_get_returns_none_on_hash_mismatch(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    emb = np.random.rand(256).astype(np.float32)
    cache.put("a.md", "hash1", emb)
    assert cache.get("a.md", "wrong_hash") is None
    cache.close()


def test_get_returns_none_for_missing_path(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    assert cache.get("nonexistent.md", "hash1") is None
    cache.close()


def test_put_overwrites_on_same_path(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    emb1 = np.ones(256, dtype=np.float32)
    emb2 = np.zeros(256, dtype=np.float32)
    cache.put("a.md", "hash1", emb1)
    cache.put("a.md", "hash2", emb2)
    assert cache.get("a.md", "hash1") is None
    result = cache.get("a.md", "hash2")
    assert result is not None
    np.testing.assert_array_almost_equal(result, emb2)
    cache.close()


def test_put_batch(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    entries = [
        ("a.md", "h1", np.ones(256, dtype=np.float32)),
        ("b.md", "h2", np.zeros(256, dtype=np.float32)),
    ]
    cache.put_batch(entries)
    r1 = cache.get("a.md", "h1")
    r2 = cache.get("b.md", "h2")
    assert r1 is not None
    assert r2 is not None
    np.testing.assert_array_almost_equal(r1, np.ones(256))
    np.testing.assert_array_almost_equal(r2, np.zeros(256))
    cache.close()


def test_prune(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.put("a.md", "h1", np.ones(256, dtype=np.float32))
    cache.put("b.md", "h2", np.ones(256, dtype=np.float32))
    cache.put("c.md", "h3", np.ones(256, dtype=np.float32))
    deleted = cache.prune({"a.md", "c.md"})
    assert deleted == 1
    assert cache.get("a.md", "h1") is not None
    assert cache.get("b.md", "h2") is None
    assert cache.get("c.md", "h3") is not None
    cache.close()


def test_prune_no_stale(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.put("a.md", "h1", np.ones(256, dtype=np.float32))
    deleted = cache.prune({"a.md"})
    assert deleted == 0
    cache.close()


def test_serialization_roundtrip(tmp_path: Path) -> None:
    """Ensure float32 fidelity through blob serialization."""
    cache = _make_cache(tmp_path)
    emb = np.array([1.0, -1.0, 0.5, 1e-7, 3.14], dtype=np.float32)
    cache.put("x.md", "h", emb)
    result = cache.get("x.md", "h")
    assert result is not None
    np.testing.assert_array_equal(result, emb)
    cache.close()
