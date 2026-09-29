#!/bin/bash
# verify6.sh -- Muse alternate checkpoints (compressed-tensors MXFP4; asym
# group-32 INT4) + the gates the importer changes must not move.  gpu0 only.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/verify6-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
STORY="Write a detailed story about a lighthouse keeper."
LONG="$(cat real4k_ascii.txt)"
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh v6-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-v6-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-v6-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
echo "== Muse-Glimmer-30B-MXFP4 (compressed-tensors mxfp4-pack-quantized FFN)"
run mx Muse-Glimmer-30B-MXFP4 int4 64 "$SHORT"; echo "    text: $(txt $OUT/mx.log | tr '\n' ' ' | cut -c1-200)"
grep -hE "segfault|load: " "$OUT/mx.log" | head -2
echo "== Muse-Glimmer-30B-GPTQ-INT4 (asym, group 32 -> re-quantized g128)"
run gq Muse-Glimmer-30B-GPTQ-INT4 int4 64 "$SHORT" "GRIMOIRE_PRINT_IDS=1"; echo "    text: $(txt $OUT/gq.log | tr '\n' ' ' | cut -c1-200)"
grep -h "greedy ids" "$OUT/gq.log" | cut -c1-120
echo "== gates"
run mu Muse-Glimmer-30B-INT4-W4A16 int4 160 "$STORY"
cmp -s <(txt $OUT/mu.log) <(txt /mnt/storage/isos/grimoire-runs/grim-mudr-160.log) && echo "    Muse INT4-W4A16 160-token text IDENTICAL to fed1543" || echo "    Muse INT4-W4A16 text DIFFERS"
run w Qwen3.8-27B-W4A16 int4 64 "$SHORT"
D5=$(ls -td /mnt/storage/isos/grimoire-runs/verify5-* | head -1)
cmp -s <(txt $OUT/w.log) <(txt $D5/ws.log) && echo "    Qwen W4A16 text IDENTICAL to f8b7198" || echo "    Qwen W4A16 text DIFFERS: $(txt $OUT/w.log | tr '\n' ' ' | cut -c1-120)"
run qc1 Qwen3.8-27B-MXFP4-GRIMOIRE mxfp4 24 "$LONG"
cmp -s <(txt $OUT/qc1.log) ref-sherlock-5987-n24.txt && echo "    Sherlock n=24: IDENTICAL to reference" || echo "    Sherlock n=24: DIFFERS"
echo "ALL DONE"
