#!/usr/bin/env python
"""Score human relevance labels for a two-model recommendation comparison.

`relevance_labels.json` records, per source article, which of the recommended
articles a human judged genuinely related. The recommendations themselves are
not stored there — they are recovered by re-running the two variants, so the
labels stay meaningful only for the settings they were collected under.

    uv run --extra embedding python judgments/score_judgments.py \
        ../chezo.uno/content/post --language ja --permalink-base /post \
        --a "language=ja,prefix=トピック: ,max_content_chars=8000" \
        --b "model_name=hotchpotch/bekko-embedding-v1-a25m,\
model_file=onnx/model.onnx,pooling=mean,max_content_chars=8000"

Reports precision@k under human judgment for both variants, a paired bootstrap
interval on the difference, and — the reason this script exists — how well tag
agreement predicts the human labels. On the run these labels came from, tag
agreement scored precision 0.76 but recall 0.09: a narrow but honest slice of
relevance, which is why it could not rank the two models.

Labels were collected blind: model identity, tags and tag-hit marks hidden,
and the two columns shuffled per article. They do not inherit the tag metric's
bias, which is what makes this comparison worth anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import AbstractContextManager, nullcontext
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_embedding_variants import (  # noqa: E402
    cache_db_for,
    run_variant,
    split_front_matter,
    tags_of,
)

from prelims_cli.embedding.recommender import EmbeddingRecommender  # noqa: E402

LABELS = Path(__file__).resolve().parent / "relevance_labels.json"


def bootstrap(diffs: np.ndarray, n: int, seed: int = 0) -> tuple[float, float]:
    """95% interval on the mean per-article difference, resampling articles."""
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5]) * 100
    return float(low), float(high)


def sign_test(wins_a: int, wins_b: int) -> float:
    decided = wins_a + wins_b
    if not decided:
        return 1.0
    tail = sum(comb(decided, k) for k in range(min(wins_a, wins_b) + 1))
    return min(2 * tail / 2**decided, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_dir", type=Path)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--permalink-base", default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--a", required=True, help="config spec for the current model")
    parser.add_argument("--b", required=True, help="config spec for the candidate")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--tag-keys", nargs="+", default=["tags", "categories"])
    parser.add_argument("--ignore", nargs="*", default=["_index.md"])
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()
    # run_variant and cache_db_for read this; the labels only exist for --vary config.
    args.vary = "config"

    labels = json.loads(args.labels.read_text())
    paths = sorted(
        p
        for p in args.content_dir.rglob("*.md")
        if p.name not in args.ignore and not p.name.startswith(".")
    )
    if len(paths) < 2:
        parser.error(f"need at least 2 articles under {args.content_dir}")
    parsed = {p: split_front_matter(p) for p in paths}
    bodies = {p: body for p, (_, body) in parsed.items()}

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_home: AbstractContextManager[str] = nullcontext(str(args.cache_dir))
    else:
        cache_home = tempfile.TemporaryDirectory()
    with cache_home as home:
        before, _ = run_variant(
            paths, bodies, args, args.a, cache_db_for(home, args, args.a)
        )
        after, _ = run_variant(
            paths, bodies, args, args.b, cache_db_for(home, args, args.b)
        )

    # run_variant keys recommendations by source file path but lists them as
    # permalinks; the labels use permalinks on both ends, so re-key the sources.
    mapper = EmbeddingRecommender(
        permalink_base=args.permalink_base, language=args.language
    )
    to_permalink = {str(p): mapper._path_to_permalink(p) for p in paths}
    a = {to_permalink[k]: v for k, v in before.items()}
    b = {to_permalink[k]: v for k, v in after.items()}
    tags = {
        mapper._path_to_permalink(p): tags_of(meta, args.tag_keys)
        for p, (meta, _) in parsed.items()
    }

    judged = [p for p in a if p in labels and labels[p]["done"]]
    if not judged:
        print(
            "no labels match this corpus. The labels are keyed by permalink, so "
            "check --permalink-base and that the corpus is the one they came from."
        )
        return 1

    hits_a = hits_b = slots = wins_a = wins_b = 0
    diffs = []
    for p in judged:
        relevant = set(labels[p]["rel"])
        na, nb = len(set(a[p]) & relevant), len(set(b[p]) & relevant)
        hits_a += na
        hits_b += nb
        slots += len(a[p])
        diffs.append((nb - na) / len(a[p]))
        wins_b += nb > na
        wins_a += na > nb

    print(f"{len(judged)} judged articles of {len(a)}, {slots} recommendation slots")
    if len(judged) < len(labels):
        missing = len(labels) - len(judged)
        print(f"  ({missing} labelled articles are not in this corpus)")
    print(f"\n{'variant':<16}{'precision@' + str(args.topk):>14}{'hits':>7}")
    print(f"{'a (current)':<16}{hits_a / slots:>14.3f}{hits_a:>7}")
    print(f"{'b (candidate)':<16}{hits_b / slots:>14.3f}{hits_b:>7}")
    low, high = bootstrap(np.array(diffs), args.bootstrap)
    print(
        f"  b - a: {(hits_b - hits_a) / slots * 100:+.1f}pt, "
        f"95% CI [{low:+.1f}, {high:+.1f}] (paired bootstrap over source articles)"
    )
    print(
        f"  per-article: b {wins_b} wins / a {wins_a} wins / "
        f"{len(judged) - wins_a - wins_b} ties, "
        f"sign test p = {sign_test(wins_a, wins_b):.3f}"
    )

    # The point of the exercise: is tag agreement a usable stand-in for a human?
    tp = fp = fn = tn = 0
    blind_spot = blind_spot_relevant = 0
    for p in judged:
        relevant = set(labels[p]["rel"])
        for rec in set(a[p]) | set(b[p]):
            source_tags, rec_tags = tags.get(p, set()), tags.get(rec, set())
            human = rec in relevant
            if not source_tags or not rec_tags:
                blind_spot += 1
                blind_spot_relevant += human
            agrees = bool(source_tags & rec_tags)
            tp += agrees and human
            fp += agrees and not human
            fn += (not agrees) and human
            tn += (not agrees) and not human

    total = tp + fp + fn + tn
    print(f"\ntag agreement as a predictor of the human labels ({total} pairs)")
    print(f"  tag hit  & human relevant    {tp:>5}")
    print(f"  tag hit  & human unrelated   {fp:>5}")
    print(f"  tag miss & human relevant    {fn:>5}")
    print(f"  tag miss & human unrelated   {tn:>5}")
    if tp + fn:
        precision = tp / (tp + fp) if tp + fp else float("nan")
        print(f"  recall {tp / (tp + fn):.3f}   precision {precision:.3f}")
    print(
        f"  pairs with an untagged end, invisible to the metric: {blind_spot} "
        f"({blind_spot / total:.0%}), of which {blind_spot_relevant} "
        "were judged relevant"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
