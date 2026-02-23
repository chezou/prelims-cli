from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prelims_cli.embedding.inference import LANGUAGE_MODELS
from prelims_cli.embedding.recommender import EmbeddingRecommender, _content_hash


def _make_post(path: str, content: str) -> MagicMock:
    post = MagicMock()
    post.path = Path(path)
    post.content = content
    return post


def _fake_embeddings(texts: list[str]) -> np.ndarray:
    """Generate deterministic embeddings based on text length for testing."""
    embs = []
    for t in texts:
        seed = sum(ord(c) for c in t) % 1000
        rng2 = np.random.RandomState(seed)
        v = rng2.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        embs.append(v)
    return np.array(embs)


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_process_basic(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """Basic process: 3 posts, verify recommendations are set."""
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    posts = [
        _make_post("/posts/a.md", "Python machine learning"),
        _make_post("/posts/b.md", "Python deep learning"),
        _make_post("/posts/c.md", "Cooking recipes for dinner"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=2,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec.process(posts)

    for post in posts:
        post.update_all.assert_called_once()
        call_args = post.update_all.call_args[0]
        assert "recommendations" in call_args[0]
        assert len(call_args[0]["recommendations"]) == 2


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_permalink_generation(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    posts = [
        _make_post("/posts/MyPost.md", "content alpha"),
        _make_post("/posts/other.md", "content beta"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        lower_path=True,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec.process(posts)

    # Check that recommendations use lowercased permalinks
    for post in posts:
        recs = post.update_all.call_args[0][0]["recommendations"]
        for r in recs:
            assert r == r.lower()
            assert r.startswith("/blog/")
            assert r.endswith("/")


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_permalink_index_file(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    posts = [
        _make_post("/posts/my-article/index.md", "content one"),
        _make_post("/posts/other.md", "content two"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec.process(posts)

    # The index.md post should recommend using parent dir name "my-article"
    recs_for_other = posts[1].update_all.call_args[0][0]["recommendations"]
    assert "/blog/my-article/" in recs_for_other


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_caching_behavior(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """Second run should use cache and not call embedder."""
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    posts = [
        _make_post("/posts/a.md", "same content"),
        _make_post("/posts/b.md", "same content two"),
    ]

    cache_path = str(tmp_path / "cache.db")
    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=cache_path,
    )

    # First run: should call embedder
    rec.process(posts)
    assert MockEmbedder.called

    # Reset mock
    MockEmbedder.reset_mock()
    embedder_instance.embed.reset_mock()

    # Second run with same content: should NOT instantiate embedder
    posts2 = [
        _make_post("/posts/a.md", "same content"),
        _make_post("/posts/b.md", "same content two"),
    ]
    rec2 = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=cache_path,
    )
    rec2.process(posts2)
    MockEmbedder.assert_not_called()


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_partial_cache_hit(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """When some posts are cached and some aren't, only uncached are computed."""
    call_count = 0

    def tracking_embed(texts: list[str]) -> np.ndarray:
        nonlocal call_count
        call_count += len(texts)
        return _fake_embeddings(texts)

    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = tracking_embed

    cache_path = str(tmp_path / "cache.db")

    # First run: 2 posts
    posts1 = [
        _make_post("/posts/a.md", "content A"),
        _make_post("/posts/b.md", "content B"),
    ]
    rec = EmbeddingRecommender(permalink_base="/blog", topk=1, cache_db=cache_path)
    rec.process(posts1)
    # Reset
    call_count = 0
    MockEmbedder.reset_mock()

    # Second run: 2 cached + 1 new
    posts2 = [
        _make_post("/posts/a.md", "content A"),
        _make_post("/posts/b.md", "content B"),
        _make_post("/posts/c.md", "content C"),
    ]
    rec2 = EmbeddingRecommender(permalink_base="/blog", topk=1, cache_db=cache_path)
    rec2.process(posts2)
    # Only 1 new post should be embedded
    assert call_count == 1


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_skip_with_single_post(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """Should skip processing with fewer than 2 posts."""
    posts = [_make_post("/posts/a.md", "content")]
    rec = EmbeddingRecommender(cache_db=str(tmp_path / "cache.db"))
    rec.process(posts)
    MockEmbedder.assert_not_called()
    posts[0].update_all.assert_not_called()


def test_content_hash_deterministic() -> None:
    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    h3 = _content_hash("different")
    assert h1 == h2
    assert h1 != h3


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_content_truncation(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """Content should be truncated to max_content_chars before embedding."""
    embedded_texts: list[list[str]] = []

    def capture_embed(texts: list[str]) -> np.ndarray:
        embedded_texts.append(texts)
        return _fake_embeddings(texts)

    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = capture_embed

    long_content = "A" * 5000
    posts = [
        _make_post("/posts/a.md", long_content),
        _make_post("/posts/b.md", "short"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=str(tmp_path / "cache.db"),
        max_content_chars=100,
    )
    rec.process(posts)

    # The text passed to the embedder should be truncated
    assert len(embedded_texts) == 1
    batch = embedded_texts[0]
    assert len(batch[0]) == 100
    assert len(batch[1]) == len("short")


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_content_truncation_cache_key(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    """Cache key should reflect truncated content, not original."""
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    cache_path = str(tmp_path / "cache.db")

    posts = [
        _make_post("/posts/a.md", "A" * 5000),
        _make_post("/posts/b.md", "short"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=cache_path,
        max_content_chars=100,
    )
    rec.process(posts)
    MockEmbedder.reset_mock()
    embedder_instance.embed.reset_mock()

    # Same original content, same truncation → cache hit
    posts2 = [
        _make_post("/posts/a.md", "A" * 5000),
        _make_post("/posts/b.md", "short"),
    ]
    rec2 = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        cache_db=cache_path,
        max_content_chars=100,
    )
    rec2.process(posts2)
    MockEmbedder.assert_not_called()


@patch("prelims_cli.embedding.recommender.OnnxEmbedder")
def test_lower_path_false(MockEmbedder: MagicMock, tmp_path: Path) -> None:
    embedder_instance = MockEmbedder.return_value
    embedder_instance.embed.side_effect = _fake_embeddings

    posts = [
        _make_post("/posts/MyPost.md", "content alpha"),
        _make_post("/posts/Other.md", "content beta"),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=1,
        lower_path=False,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec.process(posts)

    # Post 0 recommends Post 1 or vice versa, paths should preserve case
    recs_0 = posts[0].update_all.call_args[0][0]["recommendations"]
    recs_1 = posts[1].update_all.call_args[0][0]["recommendations"]
    all_recs = recs_0 + recs_1
    # At least one should have uppercase
    assert any("MyPost" in r or "Other" in r for r in all_recs)


def test_language_default_uses_english() -> None:
    """Default (no language) should resolve to English granite model."""
    rec = EmbeddingRecommender(permalink_base="/blog")
    assert rec.model_name == LANGUAGE_MODELS["en"]["model_name"]
    assert rec.model_file == LANGUAGE_MODELS["en"]["model_file"]
    assert rec.cache_db == ".prelims_embedding_cache_en.db"


def test_language_en_resolves_granite() -> None:
    rec = EmbeddingRecommender(permalink_base="/blog", language="en")
    assert rec.model_name == LANGUAGE_MODELS["en"]["model_name"]
    assert "granite" in rec.model_name


def test_language_ja_resolves_ruri() -> None:
    rec = EmbeddingRecommender(permalink_base="/blog", language="ja")
    assert rec.model_name == LANGUAGE_MODELS["ja"]["model_name"]
    assert "ruri" in rec.model_name
    assert rec.cache_db == ".prelims_embedding_cache_ja.db"


def test_explicit_model_overrides_language() -> None:
    rec = EmbeddingRecommender(
        permalink_base="/blog",
        language="ja",
        model_name="custom/model-ONNX",
        model_file="onnx/custom.onnx",
    )
    assert rec.model_name == "custom/model-ONNX"
    assert rec.model_file == "onnx/custom.onnx"


def test_invalid_language_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        EmbeddingRecommender(permalink_base="/blog", language="zz")
