"""Integration test that uses the real ONNX model.

Requires: uv sync --extra embedding
Run with: uv run pytest tests/test_integration_embedding.py -m slow
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

slow = pytest.mark.slow

ML_TEXT = "Pythonで機械学習モデルを構築する方法について解説します。"
DL_TEXT = "Pythonでディープラーニングを実装するチュートリアルです。"
COOK_TEXT = "今日の晩ご飯のレシピを紹介します。簡単な料理です。"

EN_ML_TEXT = "How to build a machine learning model in Python, step by step."
EN_DL_TEXT = "A tutorial on implementing deep learning models with Python."
EN_COOK_TEXT = "Tonight's dinner recipe: a simple dish anyone can cook."


@slow
@pytest.mark.parametrize(
    "language,texts",
    [
        ("ja", (ML_TEXT, DL_TEXT, COOK_TEXT)),
        ("en", (EN_ML_TEXT, EN_DL_TEXT, EN_COOK_TEXT)),
    ],
)
def test_end_to_end_with_real_model(
    language: str, texts: tuple[str, str, str], tmp_path: Path
) -> None:
    """Each language runs its own model and its own pooling method."""
    from prelims_cli.embedding.recommender import EmbeddingRecommender

    def make_post(name: str, content: str) -> MagicMock:
        post = MagicMock()
        post.path = Path(f"/posts/{name}.md")
        post.content = content
        return post

    ml_text, dl_text, cook_text = texts

    posts = [
        make_post("python-ml", ml_text),
        make_post("python-dl", dl_text),
        make_post("cooking", cook_text),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=2,
        language=language,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec.process(posts)

    for post in posts:
        post.update_all.assert_called_once()
        recs = post.update_all.call_args[0][0]["recommendations"]
        assert len(recs) == 2
        assert all(r.startswith("/blog/") for r in recs)

    # Python ML and DL should recommend each other as top-1
    recs_ml = posts[0].update_all.call_args[0][0]["recommendations"]
    recs_dl = posts[1].update_all.call_args[0][0]["recommendations"]
    assert recs_ml[0] == "/blog/python-dl/"
    assert recs_dl[0] == "/blog/python-ml/"

    # Second run should hit cache
    posts2 = [
        make_post("python-ml", ml_text),
        make_post("python-dl", dl_text),
        make_post("cooking", cook_text),
    ]
    rec2 = EmbeddingRecommender(
        permalink_base="/blog",
        topk=2,
        language=language,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec2.process(posts2)


@slow
def test_pooling_changes_english_embeddings() -> None:
    """CLS and mean pooling must produce different vectors for the same text.

    Guards the fix itself: if the pooling setting were ignored, the English
    model would keep returning the mean-pooled vectors it used to.
    """
    import numpy as np

    from prelims_cli.embedding.inference import LANGUAGE_MODELS, OnnxEmbedder

    spec = LANGUAGE_MODELS["en"]
    texts = [EN_ML_TEXT, EN_DL_TEXT, EN_COOK_TEXT]

    cls_embs = OnnxEmbedder(
        model_name=spec["model_name"], model_file=spec["model_file"], pooling="cls"
    ).embed(texts)
    mean_embs = OnnxEmbedder(
        model_name=spec["model_name"], model_file=spec["model_file"], pooling="mean"
    ).embed(texts)

    assert cls_embs.shape == mean_embs.shape
    assert not np.allclose(cls_embs, mean_embs)
