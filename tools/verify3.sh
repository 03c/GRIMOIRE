#!/bin/bash
# verify3.sh -- batched GEMV with R rows per sub-group: verify-pass times and
# bit-identity (R never changes a row's accumulation).  gpu0 only.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/root/verify3-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
STORY="Write a detailed story about a lighthouse keeper."
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh v3-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-v3-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-v3-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$OUT/$1.log" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$OUT/$1.log" | tail -1 | cut -d= -f2; }
same() { cmp -s <(txt $1) <(txt $2) && echo IDENTICAL || echo differs; }
spec() { grep -hE "spec:|spec time" "$OUT/$1.log" | sed "s/^/    [$1] /"; }
QM=Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16
MTP="GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_SPEC_TIME=1"
echo "== MTP verify, batched GEMV R=4 (new default) vs R=1 (old)"
run m4 $QM mxfp4 160 "$STORY" "$MTP"
run m1 $QM mxfp4 160 "$STORY" "$MTP
B70_GEMV_BATCH_R=1"
echo "    text R=4 vs R=1: $(same m4 m1)   elapsed $(el m4) vs $(el m1) s"; spec m4; spec m1
Q=Qwen3.8-27B-MXFP4-GRIMOIRE
echo "== DFlash2 verify (M=16)"
run f4 $Q mxfp4 160 "$STORY" "GRIMOIRE_DFLASH_MODEL=/models/Qwen3.8-27B-DFlash2
GRIMOIRE_SPEC_TIME=1"
echo "    Qwen DFlash2 elapsed $(el f4) s (was 26.733)"; spec f4
run o4 Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 160 "$STORY" "GRIMOIRE_DFLASH_MODEL=/models/Ornith-1.5-35B-A3B-DFlash2
GRIMOIRE_SPEC_TIME=1"
echo "    Ornith DFlash2 elapsed $(el o4) s (was 6.234)"; spec o4
echo "== plain decode unchanged (MB=1 path untouched)"
run p $Q mxfp4 160 "$STORY"
echo "    plain elapsed $(el p) s (sweep: 6.339)"
echo "ALL DONE"
