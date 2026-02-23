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


@slow
def test_end_to_end_with_real_model(tmp_path: Path) -> None:
    from prelims_cli.embedding.recommender import EmbeddingRecommender

    def make_post(name: str, content: str) -> MagicMock:
        post = MagicMock()
        post.path = Path(f"/posts/{name}.md")
        post.content = content
        return post

    posts = [
        make_post("python-ml", ML_TEXT),
        make_post("python-dl", DL_TEXT),
        make_post("cooking", COOK_TEXT),
    ]

    rec = EmbeddingRecommender(
        permalink_base="/blog",
        topk=2,
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
        make_post("python-ml", ML_TEXT),
        make_post("python-dl", DL_TEXT),
        make_post("cooking", COOK_TEXT),
    ]
    rec2 = EmbeddingRecommender(
        permalink_base="/blog",
        topk=2,
        cache_db=str(tmp_path / "cache.db"),
    )
    rec2.process(posts2)
