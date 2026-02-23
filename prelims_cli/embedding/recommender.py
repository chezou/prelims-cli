from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
from prelims.processor.base import BaseFrontMatterProcessor  # type: ignore

from .cache import EmbeddingCache
from .inference import DEFAULT_LANGUAGE, LANGUAGE_MODELS, OnnxEmbedder

logger = logging.getLogger(__name__)


class EmbeddingRecommender(BaseFrontMatterProcessor):
    """Generate article recommendations using ONNX embedding model similarity.

    Model is selected by ``language`` (default ``"en"``).
    Embeddings are cached in a per-language SQLite DB keyed by content hash.

    Parameters
    ----------
    permalink_base : str
        Non-file-name part of article permalinks.
    topk : int
        Number of recommended articles (default 3).
    lower_path : bool
        Lowercase paths in recommendations (default True).
    cache_db : str | None
        SQLite database path for embedding cache.
        Defaults to ``.prelims_embedding_cache_{language}.db``.
    model_name : str | None
        HuggingFace repo ID for the ONNX model.
        If None, resolved from ``language``.
    model_file : str | None
        Path to the ONNX model file within the repo.
        If None, resolved from ``language``.
    language : str
        Language shorthand (``"en"`` or ``"ja"``). Determines the default
        model and cache DB name. Ignored when ``model_name`` is set explicitly.
    prefix : str
        Text prefix prepended to each article before embedding.
    batch_size : int
        Batch size for embedding inference.
    max_content_chars : int
        Truncate post content to this many characters before embedding.
        Reduces peak memory by limiting token sequence length.
    """

    def __init__(
        self,
        permalink_base: str = "",
        topk: int = 3,
        lower_path: bool = True,
        cache_db: str | None = None,
        model_name: str | None = None,
        model_file: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        prefix: str = "",
        batch_size: int = 8,
        max_content_chars: int = 2000,
    ) -> None:
        if model_name is None:
            if language not in LANGUAGE_MODELS:
                raise ValueError(
                    f"Unsupported language: {language!r}. "
                    f"Supported: {sorted(LANGUAGE_MODELS)}"
                )
            model_name = LANGUAGE_MODELS[language]["model_name"]
            model_file = LANGUAGE_MODELS[language]["model_file"]
        elif model_file is None:
            model_file = "onnx/model_quantized.onnx"

        self.permalink_base = permalink_base
        self.topk = topk
        self.lower_path = lower_path
        self.cache_db = cache_db or f".prelims_embedding_cache_{language}.db"
        self.model_name = model_name
        self.model_file = model_file
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_content_chars = max_content_chars

    def process(self, posts: list, allow_overwrite: bool = True) -> None:  # type: ignore
        """Compute embeddings and write recommendations to frontmatter."""
        if len(posts) < 2:
            logger.warning("Need at least 2 posts for recommendations, skipping")
            return

        cache = EmbeddingCache(self.cache_db)

        try:
            self._process_with_cache(posts, cache, allow_overwrite)
        finally:
            cache.close()

    def _process_with_cache(
        self,
        posts: list,
        cache: EmbeddingCache,
        allow_overwrite: bool,
    ) -> None:
        contents = [post.content[: self.max_content_chars] for post in posts]
        paths = [str(post.path) for post in posts]
        hashes = [_content_hash(c) for c in contents]

        # Check cache
        embeddings: list[np.ndarray | None] = []
        uncached_indices: list[int] = []
        for i, (path, h) in enumerate(zip(paths, hashes)):
            cached = cache.get(path, h)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)

        n_cached = len(posts) - len(uncached_indices)
        logger.info(f"Cache hit: {n_cached}/{len(posts)}")

        # Compute missing embeddings (lazy model load)
        if uncached_indices:
            embedder = OnnxEmbedder(
                model_name=self.model_name,
                model_file=self.model_file,
                prefix=self.prefix,
            )
            uncached_texts = [contents[i] for i in uncached_indices]

            # Process in batches
            new_embeddings: list[np.ndarray] = []
            for start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[start : start + self.batch_size]
                batch_embs = embedder.embed(batch)
                new_embeddings.extend(batch_embs)

            # Fill in and cache
            cache_entries: list[tuple[str, str, np.ndarray]] = []
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                cache_entries.append((paths[idx], hashes[idx], emb))
            cache.put_batch(cache_entries)

        # Build matrix and compute similarities
        matrix = np.stack(embeddings)  # type: ignore[arg-type]
        # Vectors are L2-normalized, so dot product = cosine similarity
        similarities = matrix @ matrix.T

        # Generate recommendations
        for i in range(len(posts)):
            sim_scores = similarities[i]
            # Exclude self (similarity=1.0)
            top_indices = np.argsort(sim_scores, kind="stable")[::-1][
                1 : (self.topk + 1)
            ]
            recommend_permalinks = [
                self._path_to_permalink(posts[j].path) for j in top_indices
            ]
            posts[i].update_all(
                {"recommendations": recommend_permalinks}, allow_overwrite
            )

        # Prune deleted articles
        active = set(paths)
        pruned = cache.prune(active)
        if pruned:
            logger.info(f"Pruned {pruned} stale entries from cache")

    def _path_to_permalink(self, path: Path) -> str:
        """Convert a file path into a permalink."""
        file = path.stem
        if file == "index":
            file = path.parent.name
        if self.lower_path:
            file = file.lower()
        return urljoin(f"{self.permalink_base}/", f"{file}/")


def _content_hash(content: str) -> str:
    """SHA-256 hash of content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
