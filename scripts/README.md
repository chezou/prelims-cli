# scripts

Evaluation helpers for the embedding recommender. Not part of the package —
run them from a checkout with `uv run --extra embedding python scripts/<name>`.

## check_onnx_model.py

Checks whether an ONNX embedding model works with `OnnxEmbedder` unchanged:
prints the model's inputs and outputs, fails if it needs anything beyond
`input_ids` and `attention_mask`, then embeds probe sentences in several
languages and verifies that same-topic pairs come out closer than unrelated
ones. Run it before adding a model to `LANGUAGE_MODELS`.

```sh
uv run --extra embedding python scripts/check_onnx_model.py \
    --model-name hotchpotch/bekko-embedding-v1-a8m \
    --model-file onnx/model.onnx --pooling mean
```

## compare_embedding_variants.py

Runs the recommender twice over the same articles with two different settings
and reports what changed. Use it to decide whether a change to pooling, prefix
or the model itself is an improvement, rather than eyeballing a diff.

```sh
uv run --extra embedding python scripts/compare_embedding_variants.py \
    ../site/content/post --language ja --permalink-base /post \
    --vary prefix --a "" --b "トピック: " --summary-only
```

It reports membership changes separately from order-only ones, and — when the
front matter has `tags` or `categories` — how often recommendations share a tag
with the article they are attached to, plus how concentrated recommendations
are on a few articles. Temporary cache DBs are used, so the real ones are left
alone.
