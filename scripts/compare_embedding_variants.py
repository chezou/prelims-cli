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
  hub spread   how concentrated recommendations are on a few articles. A
               generic post that attaches to everything shows up as a high
               max in-degree and a high top-5 share. Lower is better.

Changes are split into membership changes (different articles recommended) and
order-only changes (same set, reshuffled), because the latter barely matter.

Requires: uv sync --extra embedding
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections import Counter
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import yaml

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
    """Return (front matter dict, body). Best-effort on odd files."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    _, _, rest = text.partition("\n")
    end = rest.find("\n---")
    if end == -1:
        return {}, text
    head, body = rest[:end], rest[end + len("\n---") :].lstrip()
    try:
        meta = yaml.safe_load(head)
    except yaml.YAMLError:
        return {}, body
    return (meta if isinstance(meta, dict) else {}), body


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
    persistent --cache-dir survives swapping the two, and a corpus never shares
    a file with another one (prune() would delete the other corpus's rows).
    Embeddings do not depend on topk, so a sweep reuses them all.
    """
    digest = hashlib.sha1(f"{args.vary}\0{spec}".encode()).hexdigest()[:8]
    return str(Path(home) / f"{args.content_dir.name}-{digest}.db")


def parse_config(spec: str) -> dict[str, str]:
    """Parse "model_name=x,pooling=mean" into kwargs."""
    kwargs = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        kwargs[key.strip()] = value.strip()
    return kwargs


def run_variant(
    paths: list[Path],
    bodies: dict[Path, str],
    args: argparse.Namespace,
    value: str,
    cache_db: str,
) -> dict[str, list[str]]:
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
    EmbeddingRecommender(**kwargs).process(posts)  # type: ignore[arg-type]
    return {str(p.path): p.recommendations for p in posts}


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
        before = run_variant(
            paths, bodies, args, args.a, cache_db_for(home, args, args.a)
        )
        after = run_variant(
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

    print("\nhub spread (lower is better)")
    print(f"{'variant':<24}{'max in-deg':>12}{'top-5 share':>14}")
    for value, recs in ((args.a, before), (args.b, after)):
        peak, share = hub_spread(recs)
        print(f"{label_of(value):<24}{peak:>12}{share:>13.1%}")


if __name__ == "__main__":
    main()
