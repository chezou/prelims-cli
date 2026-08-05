# judgments

Human relevance labels for the embedding recommender comparison, and the script
that scores them. Collected because the tag-based proxy metric could not rank
two models and there was no way to tell, from inside the metric, whether that
meant the models were equal or the metric was blind.

## relevance_labels.json

147 source articles from chezo.uno (143 ja / 4 en), 762 judged recommendation
pairs, 416 marked relevant. Keyed by permalink:

```json
{
  "/post/2005-03-03-sugoi/": {
    "done": true,
    "rel": ["/post/2005-07-11-w-sim/", "/post/2005-04-23-willcom/"]
  }
}
```

`done` separates the two things a missing mark can mean: `true` with an empty
`rel` is "looked at it, nothing here is related", absent is "never opened".
Conflating them would silently score unjudged articles as zero.

The recommendations are not stored — only which of them were judged relevant.
They are recovered by re-running both variants, so the labels apply to the
settings they were collected under: **8,000 characters, topk=3, ruri-v3-30m +
prefix vs bekko-embedding-v1-a25m**. The article set is the 389 articles whose
top-3 changed between those two.

Collection conditions, which are what make the labels usable:

- **Blind.** Model identity was hidden, the two columns shuffled per article,
  and tags plus tag-hit marks hidden. Tags are the metric under test, so a
  judge who can see them is no longer independent of it.
- **Each pair judged once.** A recommendation both models made was judged a
  single time and applied to both, so no pair carries contradictory labels.
- **Stratified, not pooled.** The set is a seeded random 40 plus the 64
  "total swap" articles (no shared recommendation at all) plus incidental
  others. Only the random 40 is an unbiased sample; the swap group was picked
  for maximal disagreement and will overstate any difference.

One judge (the corpus author), so there is no inter-annotator agreement figure.

## score_judgments.py

```sh
uv run --extra embedding python judgments/score_judgments.py \
    ../chezo.uno/content/post --language ja --permalink-base /post \
    --cache-dir ~/.cache/prelims-eval-filtered \
    --a "language=ja,prefix=トピック: ,max_content_chars=8000" \
    --b "model_name=hotchpotch/bekko-embedding-v1-a25m,\
model_file=onnx/model.onnx,pooling=mean,max_content_chars=8000"
```

Prints precision@k under human judgment for both variants with a paired
bootstrap interval, a per-article sign test, and a confusion matrix of tag
agreement against the human labels.

That last table is the point. On this data tag agreement scored **precision
0.85, recall 0.07**: when tags matched, the human agreed, but 387 of the 416
relevant recommendations had no tag overlap to find. 90% of judged pairs had an
untagged article on one end and were invisible to the metric entirely. The
proxy was not wrong, it was narrow — and the models differed mostly in the part
it could not see.

The model difference itself stayed undecided: +1.7pt [−9.2, +12.5] on the
unbiased random 40, 41 wins to 32 per article (sign test p = 0.35). Judging all
389 changed articles would tighten that to roughly [−1.8, +5.1] — still short
of a verdict, which is why collection stopped at 147.
