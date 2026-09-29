#!/bin/bash
# verify4.sh -- Muse prompt attention on the XMX flash kernel (sliding window),
# plus the Qwen regression gates the shared flash kernel must keep.  gpu0 only.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/verify4-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
LONG="$(cat real4k_ascii.txt)"
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh v4-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-v4-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-v4-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$OUT/$1.log" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
same() { cmp -s <(txt $1) <(txt $2) && echo IDENTICAL || echo differs; }
MU=Muse-Glimmer-30B-INT4-W4A16
echo "== Muse prompt attention: XMX flash (window) vs the block kernel"
run ft $MU int4 1 "$LONG" "GRIMOIRE_SANDWICH_TIME=1"
sed -n '/sandwich prefill regions/,/TOTAL/p' "$OUT/ft.log"
run fl $MU int4 24 "$LONG"
run bl $MU int4 24 "$LONG" "GRIMOIRE_SANDWICH_BLOCK_ATTN=1"
echo "    long 24-token text flash vs block: $(same fl bl)"
echo "    text: $(txt fl | tr '\n' ' ' | cut -c1-160)"
run fs $MU int4 64 "$SHORT"
run bs $MU int4 64 "$SHORT" "GRIMOIRE_SANDWICH_BLOCK_ATTN=1"
echo "    short 64-token text flash vs block: $(same fs bs)"
echo "    text: $(txt fs | tr '\n' ' ' | cut -c1-200)"
Q=Qwen3.8-27B-MXFP4-GRIMOIRE
echo "== Qwen gates (shared flash kernel)"
run qpp $Q mxfp4 1 "$(cat p4096.txt)"
run qc1 $Q mxfp4 24 "$LONG"
cmp -s <(txt qc1) ref-sherlock-5987-n24.txt && echo "    Sherlock n=24: IDENTICAL to reference" || { echo "    Sherlock n=24: DIFFERS"; diff <(txt qc1) ref-sherlock-5987-n24.txt | head -4; }
echo "ALL DONE"
