# prelims-cli

CLI for [prelims](https://github.com/takuti/prelims).

## Install

Run:

```sh
pip install prelims-cli
```

If you need Japanese tokenization, run:

```sh
pip install prelims-cli[ja]
```

## Usage

Assuming the following folder directory:

```sh
- content
|  ├── post
|  └── blog
└─ scripts
   └ config
     └ myconfig.yaml
```

where, post and blog are pages, and scripts is the place to put scripts.

Here is the example of configuration for the normal recommender:

```myconfig.yaml
handlers:
  - target_path: "content/blog"
    ignore_files:
      - _index.md
    processors:
      - type: recommender
        permalink_base: "/blog"
        tfidf_options:
          stop_words: english
          max_df: 0.95
          min_df: 2
        tokenizer: null
  - target_path: "content/post"
    ignore_files:
      - _index.md
    processors:
      - type: recommender
        permalink_base: "/post"
        tfidf_options:
          max_df: 0.95
          min_df: 2
        tokenizer:
          lang: ja
          type: sudachi
          mode: C
          dict: full
```

Here is the example of configuration for the embedding-based recommender:

```myconfig-embedding.yaml
handlers:
  - target_path: "content/blog"
    ignore_files:
      - _index.md
    processors:
      - permalink_base: "/blog"
        type: embedding_recommender
        language: en  # Use onnx-community/granite-embedding-small-english-r2-ONNX
        topk: 3
        cache_db: ".prelims_embedding_cache_en.db"
  - target_path: "content/post"
    ignore_files:
      - _index.md
    processors:
      - permalink_base: "/post"
        type: embedding_recommender
        language: ja   # Use sirasagi62/ruri-v3-30m-ONNX
        topk: 3
        cache_db: ".prelims_embedding_cache_ja.db"
```

`language` picks both the model and its pooling method (`en` uses CLS pooling,
`ja` uses mean pooling).

There is a third option, `multilingual`, which uses one model
([bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m))
for every language — useful when a site mixes languages, or writes in one the
other two do not cover:

```yaml
      - permalink_base: "/post"
        type: embedding_recommender
        language: multilingual
        topk: 3
        cache_db: ".prelims_embedding_cache.db"
```

On a 453-article Japanese/English corpus it scored the same as the per-language
pair rather than better: 162 recommendations judged by hand, blind to the model,
came out +1.7pt with a 95% interval of [-9.2, +12.5], and the English side —
judged exhaustively — tied exactly. What differs is temperament. It leans toward
linking the same entity (the same artist, author, sibling event); the
per-language models lean toward topic purity. Pick on that, and on cost: 199 MB
of fp32 weights against 37–52 MB of int8, and roughly 0.4s per article on a CPU
runner. Switching re-embeds everything once, since the model is part of the
cache key. The measurements are in [`judgments/`](judgments/).

Give each handler its own `cache_db` even when they share a model: `prune()`
deletes rows for articles a handler does not see, so a shared file would have
each handler wiping the other's.

When you point the recommender at another model with `model_name`, you must also
state its `pooling` (`mean` or `cls`) — check the model card, because the wrong
pooling degrades the embeddings without raising an error:

```yaml
      - permalink_base: "/blog"
        type: embedding_recommender
        model_name: "your-org/your-model-ONNX"
        model_file: "onnx/model_quantized.onnx"
        pooling: cls
```

Model revisions are pinned to a commit in `LANGUAGE_MODELS`. Without a pin,
an upstream re-upload would leave vectors from two different models in the same
cache DB, compared against each other — same dimensions, no error, quietly worse
recommendations. To take an upstream update, bump the `revision` there; the
whole cache re-embeds and stays one generation. `revision` is also accepted in
the config for a custom `model_name`.

Cached embeddings are keyed by the model, its revision, pooling and prefix as
well as the article content, so changing any of them re-embeds the affected
articles instead of reusing stale vectors. Changes to the embedding code itself are not
visible in those settings, so `EMBEDDING_CACHE_VERSION` in
`prelims_cli/embedding/inference.py` is part of the key too — bump it in any
change that makes `embed()` return different vectors for the same input.


```sh
$ prelims-cli --config ./scripts/config/myconfig.yaml
target: /user/chezo/src/chezo.uno/content/blog
target: /users/chezo/src/chezo.uno/content/post
```

Then your articles' front matter were updated.

Articles whose front matter did not change are left untouched, and updated ones
are written in block style:

```yaml
tags:
  - AI
  - Hugo
```

This keeps the output stable across runs, so a CMS or an editor writing the same
style does not produce a diff on every run.
