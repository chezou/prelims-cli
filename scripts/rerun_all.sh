#!/usr/bin/env bash
#
# Regenerate every number in the embedding evaluation report.
#
#   bash scripts/rerun_all.sh
#
# Embeddings are keyed on article text only, so a change to front matter tags
# does not invalidate them: with a warm --cache-dir every run below returns
# immediately. Output goes to both the terminal and $OUT.
#
set -uo pipefail

CACHE="${CACHE:-$HOME/.cache/prelims-eval-filtered}"
OUT="${OUT:-./norm-run.txt}"
JA="${JA:-../chezo.uno/content/post}"
EN="${EN:-../chezo.uno/content/blog}"

RURI="language=ja,prefix=トピック: "
GRAN="language=en"
A8M="model_name=hotchpotch/bekko-embedding-v1-a8m,model_file=onnx/model.onnx,pooling=mean"
A25M="model_name=hotchpotch/bekko-embedding-v1-a25m,model_file=onnx/model.onnx,pooling=mean"
G97="model_name=onnx-community/granite-embedding-97m-multilingual-r2-ONNX,pooling=cls"
G97F="$G97,model_file=onnx/model.onnx"
G97Q="$G97,model_file=onnx/model_quantized.onnx"
RURIF="model_name=sirasagi62/ruri-v3-30m-ONNX,model_file=onnx/model.onnx,pooling=mean,prefix=トピック: "
GRANF="model_name=onnx-community/granite-embedding-small-english-r2-ONNX,model_file=onnx/model.onnx,pooling=cls"

CURATED=(--summary-only --tag-keys tags categories)
DIFF=(--tag-keys tags categories)
KEYWORDS=(--summary-only)

run() {
  local dir=$1 lang=$2 base=$3 k=$4 a=$5 b=$6
  shift 6
  uv run --extra embedding python scripts/compare_embedding_variants.py \
    "$dir" --language "$lang" --permalink-base "$base" --topk "$k" \
    --vary config --cache-dir "$CACHE" --a "$a" --b "$b" "$@"
}

ja() { run "$JA" ja /post "$@"; }
en() { run "$EN" en /blog "$@"; }

main() {
  echo "cache: $CACHE"
  echo

  for K in 3 5 10; do
    for M in "$A8M" "$A25M" "$G97F" "$G97Q"; do
      echo "===== EXP1 ja 2000 topk=$K  b=${M:0:60} ====="
      ja "$K" "$RURI,max_content_chars=2000" "$M,max_content_chars=2000" "${CURATED[@]}"
      echo "===== EXP1 en 2000 topk=$K  b=${M:0:60} ====="
      en "$K" "$GRAN,max_content_chars=2000" "$M,max_content_chars=2000" "${CURATED[@]}"
    done
  done

  for K in 3 5 10; do
    echo "===== LEN ja 2000vs8000 topk=$K ====="
    ja "$K" "$RURI,max_content_chars=2000" "$RURI,max_content_chars=8000" "${CURATED[@]}"
    echo "===== LEN en 2000vs8000 topk=$K ====="
    en "$K" "$GRAN,max_content_chars=2000" "$GRAN,max_content_chars=8000" "${CURATED[@]}"
  done

  for K in 3 5 10; do
    echo "===== INTERACT ja 8000 ruri vs a25m topk=$K ====="
    ja "$K" "$RURI,max_content_chars=8000" "$A25M,max_content_chars=8000" "${CURATED[@]}"
    echo "===== INTERACT en 8000 granite vs a25m topk=$K ====="
    en "$K" "$GRAN,max_content_chars=8000" "$A25M,max_content_chars=8000" "${CURATED[@]}"
  done

  for K in 3 5 10; do
    echo "===== QUANT ja ruri int8 vs fp32 8000 topk=$K ====="
    ja "$K" "$RURI,max_content_chars=8000" "$RURIF,max_content_chars=8000" "${CURATED[@]}"
    echo "===== QUANT en granite int8 vs fp32 8000 topk=$K ====="
    en "$K" "$GRAN,max_content_chars=8000" "$GRANF,max_content_chars=8000" "${CURATED[@]}"
  done

  echo "===== REFEREE ja keywords込み 2000 topk=5 ====="
  ja 5 "$RURI,max_content_chars=2000" "$A8M,max_content_chars=2000" "${KEYWORDS[@]}"
  echo "===== REFEREE en keywords込み 2000 topk=5 ====="
  en 5 "$GRAN,max_content_chars=2000" "$A8M,max_content_chars=2000" "${KEYWORDS[@]}"

  echo "===== DETAIL ja 8000 topk=3 per-article diff ====="
  ja 3 "$RURI,max_content_chars=8000" "$A25M,max_content_chars=8000" "${DIFF[@]}"
  echo "===== DETAIL en 8000 topk=3 per-article diff ====="
  en 3 "$GRAN,max_content_chars=8000" "$A25M,max_content_chars=8000" "${DIFF[@]}"
}

main 2>&1 | tee "$OUT"
echo "saved: $OUT" >&2
