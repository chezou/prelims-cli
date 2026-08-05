# judgments

Human relevance labels for the embedding recommender comparison, and the script
that scores them. Collected because the tag-based proxy metric could not rank
two models and there was no way to tell, from inside the metric, whether that
meant the models were equal or the metric was blind.

## relevance_labels.json

162 source articles from chezo.uno (143 ja / 19 en), 827 judged recommendation
pairs, 458 marked relevant. Japanese is a sample; English is every article whose
recommendations changed. Keyed by permalink:

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

## report.html, report.en.html

The full writeup these labels belong to — the model comparison, the metric that
had to be repaired mid-way, the human evaluation, and how the decision was
actually made. `report.html` is the Japanese original and `report.en.html` an
English translation of it; both are self-contained, so open either in a browser.

Read it before reusing any number from here, because most of the report is about
why the obvious readings of those numbers are wrong.

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
0.76, recall 0.09**: 418 of the 458 relevant recommendations had no tag overlap
to find, and 86% of judged pairs had an untagged article on one end, invisible
to the metric entirely. The proxy was not wrong, it was narrow — and the models
differed mostly in the part it could not see.

It breaks differently per language. Japanese: precision 0.86, recall 0.06 —
right when it fires, almost blind otherwise. English, where 70% of articles
carry tags rather than 23%: recall 0.29 but precision 0.63, so tag agreement
becomes loose enough to call unrelated pairs related. Neither end of that
trade-off makes it a stand-in for a human.

The model difference itself stayed undecided: +1.7pt [−9.2, +12.5] on the
unbiased random 40, 46 wins to 36 per article (sign test p = 0.32). Judging all
389 changed articles would tighten that to roughly [−1.8, +5.1] — still short of
a verdict, which is why collection stopped at 162.

English is the one place a verdict was reachable, because all 19 changed
articles were judged: **40/57 against 40/57, an exact tie**. That matters beyond
English, because the fixed-pool precision@3 on the same corpus had put the
candidate ahead by +5.8pt [+1.4, +11.6] — the only interval in the whole study
that excluded zero. Human judgment did not reproduce it. Treat a lone
significant cell on 23 tagged articles as what it is.
