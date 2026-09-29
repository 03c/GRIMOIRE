#!/bin/bash
# verify5.sh -- group-64 INT4 (Qwen3.8-27B-W4A16) + regressions on group-128 INT4
# (Muse) and MXFP4 (Qwen Sherlock).  gpu0 only.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/verify5-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
STORY="Write a detailed story about a lighthouse keeper."
LONG="$(cat real4k_ascii.txt)"
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh v5-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-v5-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-v5-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$OUT/$1.log" | tail -1 | cut -d= -f2; }
tps() { python3 -c "a=float('$(el $1)');b=float('$(el $2)');per=(b-a)/$3;print(f'{per*1000:.2f} ms/token = {1/per:.1f} tok/s')"; }
W=Qwen3.8-27B-W4A16
echo "== Qwen3.8-27B-W4A16 (AutoRound GPTQ, group 64)"
run ws $W int4 64 "$SHORT"; echo "    text: $(txt $OUT/ws.log | tr '\n' ' ' | cut -c1-220)"
run wl $W int4 24 "$LONG"; echo "    text: $(txt $OUT/wl.log | tr '\n' ' ' | cut -c1-200)"
run w32 $W int4 32 "$STORY"; run w160 $W int4 160 "$STORY"; echo "    decode: $(tps w32 w160 128)"
echo "== Muse (INT4 group 128): must be byte-identical to fed1543 (mudr-160)"
run mu Muse-Glimmer-30B-INT4-W4A16 int4 160 "$STORY"
cmp -s <(txt $OUT/mu.log) <(txt /mnt/storage/isos/grimoire-runs/grim-mudr-160.log) && echo "    Muse 160-token text IDENTICAL to fed1543" || echo "    Muse text DIFFERS from fed1543"
echo "== Qwen MXFP4 Sherlock gate"
run qc1 Qwen3.8-27B-MXFP4-GRIMOIRE mxfp4 24 "$LONG"
cmp -s <(txt $OUT/qc1.log) ref-sherlock-5987-n24.txt && echo "    Sherlock n=24: IDENTICAL to reference" || echo "    Sherlock n=24: DIFFERS"
echo "ALL DONE"
