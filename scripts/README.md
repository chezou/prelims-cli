# scripts

Evaluation helpers for the embedding recommender. Not part of the package —
run them from a checkout with `uv run --extra embedding python scripts/<name>`.

## check_onnx_model.py

Checks whether an ONNX embedding model works with `OnnxEmbedder` unchanged:
prints the model's inputs and outputs, fails if it needs anything beyond
`input_ids` and `attention_mask`, then embeds probe sentences in several
languages and verifies that same-topic pairs come out closer than unrelated
ones. Run it before adding a model to `LANGUAGE_MODELS`.

It also reports the model's positional limit and how many characters that comes
to in each language, which is what decides whether raising `max_content_chars`
can do anything for that model at all. `MAX_LENGTH` truncates every model at the
same 8192 tokens, so a model with a shorter window is fed past its limit — that
does not raise, it just degrades, and the script warns about it.

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
front matter has `tags`, `categories` or `keywords` — how often recommendations
share a tag with the article they are attached to, plus how concentrated
recommendations are on a few articles. Temporary cache DBs are used, so the real
ones are left alone.

Check the `tag source` lines in the output before quoting the overlap number.
If most articles are tagged only through an auto-extracted `keywords` list, the
metric drifts toward lexical overlap and flatters models that rank by surface
wording. `--tag-keys tags categories` restricts it to curated tags (at the cost
of coverage) and `--max-df 0.05` drops filler terms; running both is the way to
tell whether a result survives.
