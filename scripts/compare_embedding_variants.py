#!/usr/bin/env python
"""Diff and score the related-article output of two embedding settings.

Runs the recommender twice over the same articles — once per variant — into
throwaway cache DBs, then reports what changed and which variant looks better.
Your real cache DBs are never touched.

    # pooling
    uv run --extra embedding python scripts/compare_embedding_variants.py \
        content/blog --language en --permalink-base /blog \
        --vary pooling --a mean --b cls

    # prefix
    uv run --extra embedding python scripts/compare_embedding_variants.py \
        content/post --language ja --permalink-base /post \
        --vary prefix --a "" --b "トピック: " --summary-only

    # a different model entirely (key=value pairs passed to EmbeddingRecommender)
    uv run --extra embedding python scripts/compare_embedding_variants.py \
        content/post --language ja --permalink-base /post --topk 5 --vary config \
        --a "language=ja" \
        --b "model_name=hotchpotch/bekko-embedding-v1-a8m,\
model_file=onnx/model.onnx,pooling=mean"

Metrics, when the front matter carries tags/categories/keywords:

  tag overlap  fraction of recommendations sharing >=1 tag with the article
               they are recommended for, and the mean number of shared tags.
               Higher is better — a proxy for "topically related".
               Read the "tag source" lines before trusting it: an
               auto-extracted `keywords` list turns this into something close
               to lexical overlap, which flatters models that rank by surface
               wording. Re-run with --tag-keys tags categories to check a
               result against curated tags only, and --max-df to drop filler
               terms that match everywhere.
  judged pool  the metrics to compare models on. Tag overlap above judges
               whatever the model recommends, so the model controls its own
               denominator: recommending untagged articles is free, and two
               models are not scored against the same pairs. These two are:

               precision@k with candidates restricted to tagged articles
               (the condensed-list treatment of incomplete judgments) — a
               fixed denominator of k slots per tagged source, identical for
               every variant; and pairwise AUC — the probability that a
               same-label pair of tagged articles is ranked more similar
               than a different-label pair, which uses the full similarity
               ordering instead of a top-k cutoff and therefore resolves
               smaller differences on a small corpus. Both come with paired
               bootstrap 95% CIs on the b−a difference (--bootstrap).

  hub spread   how concentrated recommendations are on a few articles. A
               generic post that attaches to everything shows up as a high
               max in-degree and a high top-5 share. Lower is better.

Changes are split into membership changes (different articles recommended) and
order-only changes (same set, reshuffled), because the latter barely matter.

Requires: uv sync --extra embedding
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import inspect
import tempfile
from collections import Counter
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
from prelims.post import Post as PrelimsPost  # type: ignore

from prelims_cli.embedding.recommender import EmbeddingRecommender

TAG_KEYS = ("tags", "categories", "keywords")


class Post:
    """Minimal stand-in for a prelims post."""

    def __init__(self, path: Path, content: str) -> None:
        self.path = path
        self.content = content
        self.recommendations: list[str] = []

    def update_all(self, values: dict, allow_overwrite: bool = True) -> None:
        self.recommendations = values["recommendations"]


def split_front_matter(path: Path) -> tuple[dict, str]:
    """Return (front matter dict, embedded text) for one article.

    The text is what prelims itself would hand the recommender, not the raw
    markdown: prelims strips HTML tags, code fences, math and URLs before
    embedding. Reading the file directly overstates how much the model sees —
    on a corpus of Medium exports it counted 154 articles over 2000 characters
    where the real pipeline had 93 — which makes any comparison of input
    length measure something other than what runs in production.

    Parsed by prelims' own loader so the two cannot drift apart.
    """
    post = PrelimsPost.load(path)
    meta = post.front_matter if isinstance(post.front_matter, dict) else {}
    return meta, post.content


def tags_of(meta: dict, keys: Sequence[str] = TAG_KEYS) -> set[str]:
    tags: set[str] = set()
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str):
            tags.add(value.strip().lower())
        elif isinstance(value, list):
            tags.update(str(v).strip().lower() for v in value if v is not None)
    return tags


def drop_common(
    tagged: dict[Path, set[str]], max_df: float
) -> tuple[dict[Path, set[str]], list[tuple[str, int]]]:
    """Drop terms that appear in more than max_df of the tagged articles.

    Auto-extracted keyword lists carry filler terms that match everywhere and
    say nothing about topic. Returns the filtered map and what was dropped.
    """
    if max_df >= 1.0:
        return tagged, []
    df = Counter(term for terms in tagged.values() for term in terms)
    limit = max_df * len(tagged)
    dropped = sorted(
        ((t, n) for t, n in df.items() if n > limit), key=lambda kv: -kv[1]
    )
    stop = {t for t, _ in dropped}
    filtered = {p: terms - stop for p, terms in tagged.items()}
    return {p: terms for p, terms in filtered.items() if terms}, dropped


def cache_db_for(home: str, args: argparse.Namespace, spec: str) -> str:
    """Path of the cache DB for one variant.

    Named after what the variant is rather than whether it is --a or --b, so a
    persistent --cache-dir survives swapping the two. Embeddings do not depend
    on topk, so a sweep reuses them all.

    The corpus's full path is in the digest, not just its directory name: two
    checkouts can both end in content/blog, and sharing a file between them
    would let prune() delete the other corpus's rows.
    """
    corpus = args.content_dir.resolve()
    digest = hashlib.sha1(f"{corpus}\0{args.vary}\0{spec}".encode()).hexdigest()[:8]
    return str(Path(home) / f"{corpus.name}-{digest}.db")


def parse_config(spec: str) -> dict[str, object]:
    """Parse "model_name=x,pooling=mean" into kwargs.

    Values are taken verbatim: ruri-v3's prefix is "トピック: " with a trailing
    space, and stripping it silently embeds something else than the model was
    trained for.
    """
    kwargs: dict[str, object] = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair.strip()!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key in RESERVED_KEYS:
            raise ValueError(
                f"{key!r} is set by this script and cannot be varied; "
                f"use --{key.replace('_', '-')} instead"
            )
        if key not in _PARAMS:
            close = difflib.get_close_matches(key, sorted(set(_PARAMS) - {"self"}))
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(
                f"unknown setting {key!r}."
                f" Known: {sorted(set(_PARAMS) - RESERVED_KEYS - {'self'})}.{hint}"
            )
        kwargs[key] = _coerce(key, value)
    return kwargs


_PARAMS = inspect.signature(EmbeddingRecommender.__init__).parameters

# Varying these would make the report describe something other than what ran:
# topk is printed from args and drives the metrics, and permalink_base and
# lower_path decide the keys that recommendations are matched on.
RESERVED_KEYS = frozenset({"permalink_base", "topk", "lower_path", "cache_db"})


def _coerce(key: str, value: str) -> object:
    """Cast to the type of the recommender's default for that parameter.

    max_content_chars and batch_size are ints; handing them a string makes the
    slice raise deep inside process().
    """
    default = _PARAMS[key].default if key in _PARAMS else None
    if isinstance(default, bool):
        return value.strip().lower() not in ("false", "0", "no")
    if isinstance(default, int):
        return int(value)
    return value


def run_variant(
    paths: list[Path],
    bodies: dict[Path, str],
    args: argparse.Namespace,
    value: str,
    cache_db: str,
) -> tuple[dict[str, list[str]], np.ndarray]:
    kwargs: dict[str, object] = {
        "permalink_base": args.permalink_base,
        "topk": args.topk,
        "language": args.language,
        "cache_db": cache_db,
    }
    if args.vary == "config":
        kwargs.update(parse_config(value))
    else:
        kwargs[args.vary] = value

    posts = [Post(p, bodies[p]) for p in paths]
    recommender = EmbeddingRecommender(**kwargs)
    recommender.process(posts)  # type: ignore[arg-type]
    # Row i embeds paths[i]; every entry was just cached by process(), so
    # this is a cache read, not a second inference pass.
    matrix = recommender.embed_posts(posts)  # type: ignore[arg-type]
    return {str(p.path): p.recommendations for p in posts}, matrix


def tag_overlap(
    recs: dict[str, list[str]], tags_by_permalink: dict[str, set[str]]
) -> tuple[float, float, int, int]:
    """Return (fraction sharing >=1 tag, mean shared tags, articles, pairs).

    Only pairs where both articles carry tags are counted, so a corpus that is
    mostly untagged resolves far less than its article count suggests.
    """
    hits = shared = considered = scored = 0
    for path, recommended in recs.items():
        own = tags_by_permalink.get(path)
        if not own:
            continue
        scored += 1
        for permalink in recommended:
            other = tags_by_permalink.get(permalink)
            if other is None:
                continue
            considered += 1
            overlap = len(own & other)
            shared += overlap
            hits += overlap > 0
    if not considered:
        return 0.0, 0.0, scored, 0
    return hits / considered, shared / considered, scored, considered


def hub_spread(recs: dict[str, list[str]]) -> tuple[int, float]:
    """Return (max in-degree, share of all slots taken by the top 5 articles)."""
    counts = Counter(p for recommended in recs.values() for p in recommended)
    total = sum(counts.values())
    if not total:
        return 0, 0.0
    top5 = sum(c for _, c in counts.most_common(5))
    return counts.most_common(1)[0][1], top5 / total


class JudgedPool:
    """The tagged articles of a corpus, as a self-contained evaluation pool.

    The any-tag metrics above judge whatever the model happens to recommend,
    which lets the model pick its own denominator: recommending untagged
    articles is free under one reading and costly under the other, and the
    two readings can rank two models in opposite orders. Everything in this
    class fixes that by restricting both candidates and sources to the tagged
    articles, so every variant answers the same question on the same pool.
    """

    def __init__(self, paths: list[Path], tagged: dict[Path, set[str]]) -> None:
        self.indices = [i for i, p in enumerate(paths) if p in tagged]
        labels = [tagged[paths[i]] for i in self.indices]
        n = len(labels)
        self.same = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(a + 1, n):
                self.same[a, b] = self.same[b, a] = bool(labels[a] & labels[b])
        self.pair_i, self.pair_j = np.triu_indices(n, k=1)

    def __len__(self) -> int:
        return len(self.indices)

    def similarities(self, matrix: np.ndarray) -> np.ndarray:
        judged = matrix[self.indices]
        return judged @ judged.T

    def condensed_hits(self, sims: np.ndarray, topk: int) -> np.ndarray:
        """Per-source count of top-k tagged candidates sharing a label.

        The condensed-list treatment of incomplete judgments: rank only the
        judged articles, so the denominator is topk per source for every
        variant, no matter what it would have recommended in production.
        """
        sims = sims.copy()
        np.fill_diagonal(sims, -np.inf)
        hits = np.zeros(len(self), dtype=int)
        for a in range(len(self)):
            top = np.argsort(sims[a], kind="stable")[::-1][:topk]
            hits[a] = int(self.same[a, top].sum())
        return hits

    def ceiling(self, topk: int) -> int:
        """Best condensed hit count any ranking could reach."""
        partners = self.same.sum(axis=1)
        return int(np.minimum(partners, min(topk, len(self) - 1)).sum())

    def auc(self, sims: np.ndarray) -> float:
        """P(same-label pair ranks above different-label pair), ties at 0.5.

        Uses the whole similarity ordering over the pool's pairs instead of a
        top-k cutoff, so with a few hundred labelled pairs it resolves
        differences that hit counts cannot. 0.5 is chance.
        """
        return _auc(sims[self.pair_i, self.pair_j], self.same[self.pair_i, self.pair_j])

    def bootstrap_auc_diff(
        self, sims_a: np.ndarray, sims_b: np.ndarray, n_boot: int, seed: int = 0
    ) -> tuple[float, float]:
        """95% CI for auc(b) − auc(a), resampling articles, not pairs.

        Pairs sharing an article are correlated, so resampling pairs directly
        would understate the interval. Articles are resampled with
        replacement and the pair set is rebuilt from the sample; pairs whose
        two positions drew the same original article are dropped, since a
        duplicated article is trivially similar to itself.
        """
        rng = np.random.default_rng(seed)
        diffs = np.empty(n_boot)
        for t in range(n_boot):
            sample = rng.integers(0, len(self), len(self))
            oi, oj = sample[self.pair_i], sample[self.pair_j]
            keep = oi != oj
            oi, oj = oi[keep], oj[keep]
            pos = self.same[oi, oj]
            diffs[t] = _auc(sims_b[oi, oj], pos) - _auc(sims_a[oi, oj], pos)
        low, high = np.percentile(diffs[~np.isnan(diffs)], [2.5, 97.5])
        return float(low), float(high)


def _auc(sims: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _average_ranks(sims)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged — what the Mann-Whitney U statistic needs."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    return ((starts + ends) / 2.0)[inverse]


def bootstrap_hits_diff(
    hits_a: np.ndarray, hits_b: np.ndarray, topk: int, n_boot: int, seed: int = 0
) -> tuple[float, float]:
    """95% CI for the condensed precision difference, resampling sources.

    Paired: each draw takes the same sources from both variants, so the
    interval is on the difference itself, not on two noisy levels.
    """
    rng = np.random.default_rng(seed)
    per_source = (hits_b - hits_a) / topk
    n = len(per_source)
    diffs = np.array([per_source[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def label_of(value: str, width: int = 22) -> str:
    """Short table label. Full values are printed in the header line."""
    label = repr(value) if value != "" else "'' (empty)"
    return label if len(label) <= width else label[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_dir", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--permalink-base", default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--vary",
        choices=["pooling", "prefix", "config"],
        default="pooling",
        help="config: --a/--b are key=value pairs for EmbeddingRecommender",
    )
    parser.add_argument("--a", default="mean", help="baseline value (before)")
    parser.add_argument("--b", default="cls", help="candidate value (after)")
    parser.add_argument("--ignore", nargs="*", default=["_index.md"])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="keep the embedding caches here instead of throwaway DBs, so "
        "re-running — with a different --topk, say — costs no embedding. "
        "Each variant gets its own file keyed by its settings",
    )
    parser.add_argument(
        "--tag-keys",
        nargs="+",
        default=list(TAG_KEYS),
        help="front matter keys used as tags. Auto-extracted `keywords` make "
        "the metric closer to lexical overlap, which favours models that rank "
        "by surface wording — restrict to curated keys to check a result",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=1.0,
        help="drop tags appearing in more than this fraction of articles "
        "(1.0 = keep everything). Filters filler terms out of keyword lists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="print at most N per-article diffs (0 = all); the count of "
        "omitted ones is reported",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="skip the per-article diff, print only the summary and metrics",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="bootstrap resamples for the 95%% CIs on the judged-pool "
        "metrics (0 = skip the intervals)",
    )
    args = parser.parse_args()

    # Without this the spec is passed through as a pooling method and the error
    # comes from deep inside the recommender, naming neither flag.
    if args.vary != "config":
        for flag, value in (("--a", args.a), ("--b", args.b)):
            if "=" in value:
                parser.error(
                    f"{flag} is {value!r}, which looks like a config spec, but "
                    f"--vary is {args.vary!r}. Add --vary config."
                )

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
        print(f"{len(paths)} articles, varying {args.vary}: {args.a!r} -> {args.b!r}\n")
        before, matrix_a = run_variant(
            paths, bodies, args, args.a, cache_db_for(home, args, args.a)
        )
        after, matrix_b = run_variant(
            paths, bodies, args, args.b, cache_db_for(home, args, args.b)
        )

    # The recommender maps file paths to permalinks; to look up a recommended
    # article's tags we need the reverse, so recompute them the same way.
    mapper = EmbeddingRecommender(
        permalink_base=args.permalink_base, language=args.language
    )
    tagged = {
        p: tags_of(meta, args.tag_keys)
        for p, (meta, _) in parsed.items()
        if tags_of(meta, args.tag_keys)
    }
    tagged, dropped = drop_common(tagged, args.max_df)
    tags_by_permalink = {str(p): tags for p, tags in tagged.items()}
    tags_by_permalink.update(
        {mapper._path_to_permalink(p): tags for p, tags in tagged.items()}
    )

    membership = order_only = printed = omitted = 0
    for path in before:
        if before[path] == after[path]:
            continue
        same_set = set(before[path]) == set(after[path])
        if same_set:
            order_only += 1
        else:
            membership += 1
        if args.summary_only:
            continue
        if args.limit and printed >= args.limit:
            omitted += 1
            continue
        printed += 1
        print(f"{path}{'  (order only)' if same_set else ''}")
        for r in before[path]:
            print(f"  - {r}")
        for r in after[path]:
            print(f"  + {r}")
        print()

    if omitted:
        print(f"({omitted} more changed articles not shown, --limit {args.limit})\n")

    changed = membership + order_only
    print(
        f"{changed}/{len(before)} articles changed: "
        f"{membership} membership, {order_only} order only"
    )

    if not tagged:
        print(f"\nnone of {args.tag_keys} found in front matter — skipping tag overlap")
    else:
        # Which key actually carries the tags decides what the metric means, so
        # report it rather than letting one key quietly stand in for the others.
        print(f"\ntag source ({len(tagged)}/{len(paths)} articles tagged)")
        for key in args.tag_keys:
            n = sum(1 for _, (meta, _) in parsed.items() if meta.get(key))
            print(f"  {key:<12}{n:>5} articles")
        if dropped:
            shown = ", ".join(f"{t} ({n})" for t, n in dropped[:5])
            print(f"  dropped over --max-df {args.max_df}: {len(dropped)} — {shown}")

        print(f"\ntag overlap (higher is better), topk={args.topk}")
        header = f"{'variant':<24}{'any-tag':>10}{'avg shared':>13}"
        print(f"{header}{'scored':>9}{'pairs':>8}")
        for value, recs in ((args.a, before), (args.b, after)):
            frac, mean, scored, pairs = tag_overlap(recs, tags_by_permalink)
            row = f"{label_of(value):<24}{frac:>10.3f}{mean:>13.3f}"
            print(f"{row}{scored:>9}{pairs:>8}")
        print(
            "  pairs = recommendations where both articles are tagged; "
            "a difference smaller than a few pairs is noise"
        )
        print(
            "  caveat: the model picks what gets judged here — recommending "
            "an untagged article\n  costs nothing, so two models are not "
            "scored on the same denominator. The judged-pool\n  metrics "
            "below fix the pool and are the ones to compare models on."
        )

        pool = JudgedPool(paths, tagged)
        slots_per_source = min(args.topk, len(pool) - 1)
        if len(pool) < 3 or slots_per_source < 1:
            print("\ntoo few tagged articles for the judged-pool metrics")
        else:
            sims_a = pool.similarities(matrix_a)
            sims_b = pool.similarities(matrix_b)
            slots = len(pool) * slots_per_source
            ceiling = pool.ceiling(args.topk)

            print(
                f"\njudged-pool precision@{args.topk} (candidates restricted "
                f"to the {len(pool)} tagged articles;\nfixed denominator "
                f"{slots} = {len(pool)} sources x {slots_per_source} slots; "
                f"best any ranking could do: {ceiling} hits)"
            )
            print(f"{'variant':<24}{'precision':>11}{'hits':>7}")
            all_hits = {}
            for value, sims in ((args.a, sims_a), (args.b, sims_b)):
                hits = pool.condensed_hits(sims, args.topk)
                all_hits[value] = hits
                total = int(hits.sum())
                print(f"{label_of(value):<24}{total / slots:>11.3f}{total:>7}")
            diff = (all_hits[args.b].sum() - all_hits[args.a].sum()) / slots
            line = f"  b - a: {diff:+.3f}"
            if args.bootstrap:
                low, high = bootstrap_hits_diff(
                    all_hits[args.a], all_hits[args.b], args.topk, args.bootstrap
                )
                line += f", 95% CI [{low:+.3f}, {high:+.3f}] (bootstrap over sources)"
            print(line)

            n_pos = int(pool.same[pool.pair_i, pool.pair_j].sum())
            n_neg = len(pool.pair_i) - n_pos
            print(
                f"\npairwise AUC over the {len(pool.pair_i)} tagged pairs "
                f"({n_pos} same-label, {n_neg} different;\n0.500 = chance; "
                "uses the full similarity ranking, so it resolves smaller "
                "differences\nthan any top-k count and does not depend on topk)"
            )
            print(f"{'variant':<24}{'AUC':>11}")
            for value, sims in ((args.a, sims_a), (args.b, sims_b)):
                print(f"{label_of(value):<24}{pool.auc(sims):>11.3f}")
            diff = pool.auc(sims_b) - pool.auc(sims_a)
            line = f"  b - a: {diff:+.3f}"
            if args.bootstrap:
                low, high = pool.bootstrap_auc_diff(sims_a, sims_b, args.bootstrap)
                line += f", 95% CI [{low:+.3f}, {high:+.3f}] (bootstrap over articles)"
            print(line)

    print("\nhub spread (lower is better)")
    print(f"{'variant':<24}{'max in-deg':>12}{'top-5 share':>14}")
    for value, recs in ((args.a, before), (args.b, after)):
        peak, share = hub_spread(recs)
        print(f"{label_of(value):<24}{peak:>12}{share:>13.1%}")


if __name__ == "__main__":
    main()
