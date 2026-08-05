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

To compare two *models*, read the judged-pool section rather than the overlap
table. The overlap metric judges whatever each model recommends, so the model
controls its own denominator — recommending untagged articles is free — and
two models are not scored against the same pairs; precision- and recall-style
readings of it can rank the same two models in opposite orders. The judged-pool
metrics restrict both sources and candidates to the tagged articles: a
precision@k with a fixed denominator (k slots per tagged source), and a
pairwise AUC that uses the full similarity ordering — on a small corpus the
AUC resolves differences that top-k hit counts cannot. Both come with paired
bootstrap 95% CIs on the difference (`--bootstrap`, default 1000 resamples).

`rerun_all.sh` drives this script over a whole evaluation in one go.

## rerun_all.sh

Drives `compare_embedding_variants.py` over a whole evaluation in one go —
every model pair, both input lengths, both referees, and the per-article diffs
— and tees the output to a file. Written so that a report built on these
numbers can be regenerated rather than trusted.

```sh
bash scripts/rerun_all.sh
```

Run it with `bash` explicitly: it relies on arrays that a POSIX shell does not
have, and on word splitting that zsh does not do. Point it elsewhere with
environment variables:

```sh
JA=../site/content/post EN=../site/content/blog \
  CACHE=~/.cache/eval OUT=~/run.txt bash scripts/rerun_all.sh
```

Embeddings are keyed on article text alone, so editing front matter tags does
not invalidate them. With a warm `--cache-dir` the whole sweep returns in
seconds, which makes it cheap to re-measure after fixing the referee.

## Judging the tag metric itself

Everything here scores recommendations against tags. To find out what that is
worth, `../judgments/` holds human relevance labels for 162 articles and a
script that scores them — including a confusion matrix of tag agreement against
those labels. On this corpus tag agreement had precision 0.76 but recall 0.09,
so a model comparison that comes out flat here is not evidence the models are
equal. Read that README before drawing a conclusion from a small difference.
