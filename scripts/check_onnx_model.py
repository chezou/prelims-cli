#!/usr/bin/env python
"""Check whether an ONNX embedding model works with OnnxEmbedder as-is.

Answers the questions you need before adding a model to LANGUAGE_MODELS:
does it run on input_ids + attention_mask alone, what shape does it return,
and do its embeddings put related texts closer together than unrelated ones.

    uv run --extra embedding python scripts/check_onnx_model.py \
        --model-name hotchpotch/bekko-embedding-v1-a8m \
        --model-file onnx/model.onnx --pooling mean

Add --revision <sha> to check a specific commit. Exits non-zero if the model
needs inputs OnnxEmbedder does not supply, which is the case that would require
changing the embedder rather than just adding a config entry.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from prelims_cli.embedding.inference import (
    POOLING_METHODS,
    OnnxEmbedder,
    _cls_pool,
    _l2_normalize,
    _mean_pool,
)

SUPPLIED_INPUTS = {"input_ids", "attention_mask"}

# Two sentences on one topic and one on another, per language. If the model
# works, the same-topic pair is the closest neighbour in every language.
PROBES = {
    "ja": [
        "Pythonで機械学習モデルを構築する方法について解説します。",
        "Pythonでディープラーニングを実装するチュートリアルです。",
        "今日の晩ご飯のレシピを紹介します。簡単な料理です。",
    ],
    "en": [
        "How to build a machine learning model in Python, step by step.",
        "A tutorial on implementing deep learning models with Python.",
        "Tonight's dinner recipe: a simple dish anyone can cook.",
    ],
    "fr": [
        "Comment construire un modèle d'apprentissage automatique en Python.",
        "Un tutoriel sur l'implémentation du deep learning avec Python.",
        "La recette du dîner de ce soir : un plat simple à cuisiner.",
    ],
}


def describe_io(embedder: OnnxEmbedder) -> tuple[set[str], list]:
    """Print the model's inputs and outputs.

    Returns the required input names and the shape of the first output, which
    is what embed() pools over.
    """
    print("inputs:")
    required = set()
    for i in embedder.session.get_inputs():
        print(f"  {i.name:<20} {i.type:<24} {i.shape}")
        required.add(i.name)
    print("outputs:")
    outputs = embedder.session.get_outputs()
    for o in outputs:
        print(f"  {o.name:<20} {o.type:<24} {o.shape}")
    return required, list(outputs[0].shape)


def report_pooling_match(embedder: OnnxEmbedder, texts: list[str]) -> str | None:
    """Compare both pooling methods against the model's own pooled output.

    Sentence-transformers exports often ship a `sentence_embedding` output
    alongside the token states. It was produced by the model's own pooling
    module, so whichever of ours reproduces it is the one the model wants —
    which beats reading the model card.
    """
    input_ids, attention_mask = embedder._tokenize(texts)
    outputs = embedder.session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    pooled = next((o for o in outputs[1:] if o.ndim == 2), None)
    if pooled is None:
        print("no pooled output exported — pooling cannot be verified here")
        return None

    reference = _l2_normalize(pooled)
    candidates = {
        "mean": _mean_pool(outputs[0], attention_mask),
        "cls": _cls_pool(outputs[0]),
    }
    best, best_sim = None, -1.0
    for name, vectors in candidates.items():
        sim = float((_l2_normalize(vectors) * reference).sum(axis=1).mean())
        print(f"  {name:<6} vs exported sentence_embedding: cosine {sim:+.4f}")
        if sim > best_sim:
            best, best_sim = name, sim
    if best_sim < 0.99:
        print("  neither reproduces it closely — the export may pool differently")
        return None
    return best


def report_probes(embedder: OnnxEmbedder) -> bool:
    """Embed each probe set; return True if every same-topic pair wins."""
    all_passed = True
    for language, texts in PROBES.items():
        embeddings = embedder.embed(texts)
        similarities = embeddings @ embeddings.T
        related = similarities[0, 1]
        unrelated = max(similarities[0, 2], similarities[1, 2])
        passed = related > unrelated
        all_passed &= passed
        print(
            f"  {language}: related={related:+.3f} unrelated={unrelated:+.3f} "
            f"{'ok' if passed else 'FAILED — unrelated text is closer'}"
        )
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-file", default="onnx/model.onnx")
    parser.add_argument("--pooling", required=True, choices=list(POOLING_METHODS))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    print(f"{args.model_name} @ {args.revision or 'default branch'}")
    print(f"file={args.model_file} pooling={args.pooling} prefix={args.prefix!r}\n")

    embedder = OnnxEmbedder(
        model_name=args.model_name,
        model_file=args.model_file,
        pooling=args.pooling,
        revision=args.revision,
        prefix=args.prefix,
    )

    required, first_output_shape = describe_io(embedder)
    extra = required - SUPPLIED_INPUTS
    print()
    if extra:
        print(f"NOT USABLE AS-IS: model also requires {sorted(extra)}.")
        print("OnnxEmbedder only feeds input_ids and attention_mask.")
        return 1
    print("inputs are covered by OnnxEmbedder (input_ids + attention_mask)")

    # embed() pools over outputs[0] and expects (batch, seq_len, dim). An
    # export whose first output is already pooled would make _cls_pool return
    # a garbage 1-D slice instead of failing.
    if len(first_output_shape) != 3:
        print(
            f"NOT USABLE AS-IS: first output has rank {len(first_output_shape)}"
            f" {first_output_shape}, expected (batch, seq_len, dim)."
        )
        print("It looks pre-pooled; embed() would pool it a second time.")
        return 1

    embeddings = embedder.embed(PROBES["en"])
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"embedding shape: {embeddings.shape} (dim={embeddings.shape[1]})")
    print(f"L2 norms: {norms.round(4).tolist()}")

    print("\nwhich pooling does the model itself use?")
    wanted = report_pooling_match(embedder, PROBES["en"])
    if wanted and wanted != args.pooling:
        print(
            f"\nWRONG POOLING: this export pools with {wanted!r}, not {args.pooling!r}."
        )
        print("Re-run with the right one; the difference does not raise on its own.")
        return 1
    if wanted:
        print(f"  confirmed: {wanted}")

    print("\nsame-topic pair should beat the unrelated one:")
    if not report_probes(embedder):
        print("\nmodel runs but the embeddings do not separate topics.")
        print("Check the pooling method against the model card before using it.")
        return 1

    print("\nusable: add it to LANGUAGE_MODELS with the settings above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
