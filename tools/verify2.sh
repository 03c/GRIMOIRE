#!/bin/bash
# verify2.sh -- exactness probe for flash_decode_gqa, the parallel top-16
# (DFlash2 selector), and the Muse pure-mode prefill fallback.  gpu0 only.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/root/verify2-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
STORY="Write a detailed story about a lighthouse keeper."
LONG="$(cat real4k_ascii.txt)"
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh v2-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-v2-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-v2-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$OUT/$1.log" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$OUT/$1.log" | tail -1 | cut -d= -f2; }
tps() { python3 -c "a=float('$(el $1)');b=float('$(el $2)');per=(b-a)/$3;print(f'{per*1000:.2f} ms/token = {1/per:.1f} tok/s')"; }
same() { cmp -s <(txt $1) <(txt $2) && echo IDENTICAL || echo differs; }

echo "== flash_decode_gqa exactness: layer-3 attention output, first decode steps"
K2=K2-Horizon-MoVA-36B-A4B
P="GRIMOIRE_DEBUG=1
GRIMOIRE_PROBE_EXACT=1
GRIMOIRE_PROBE_LAYER=3"
run pr-new $K2 mxfp4 3 "$SHORT" "$P"
run pr-old $K2 mxfp4 3 "$SHORT" "$P
GRIMOIRE_FLASH_DECODE_OLD=1"
for f in pr-new pr-old; do echo "  -- $f"; grep -A1 "FA attn core" "$OUT/$f.log" | grep -v "^--" | head -4 | sed 's/^/    /'; done

echo "== DFlash2 selector: parallel top-16 vs the loop kernel"
Q=Qwen3.8-27B-MXFP4-GRIMOIRE
DF="GRIMOIRE_DFLASH_MODEL=/models/Qwen3.8-27B-DFlash2
GRIMOIRE_SPEC_TIME=1"
run qf-new $Q mxfp4 32 "$STORY" "$DF"
run qf-loop $Q mxfp4 32 "$STORY" "$DF
GRIMOIRE_TOPK16_LOOP=1"
echo "    32-token text parallel vs loop top-16: $(same qf-new qf-loop)"
for f in qf-new qf-loop; do grep -hE "spec:|spec time" "$OUT/$f.log" | sed "s/^/    [$f] /"; done
run qf-160 $Q mxfp4 160 "$STORY" "$DF"
echo "    Qwen DFlash2: $(tps qf-new qf-160 128)"; grep -hE "spec:|spec time" "$OUT/qf-160.log" | sed 's/^/   /'
O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
ODF="GRIMOIRE_DFLASH_MODEL=/models/Ornith-1.5-35B-A3B-DFlash2
GRIMOIRE_SPEC_TIME=1"
run of-32 $O mxfp4 32 "$STORY" "$ODF"
run of-160 $O mxfp4 160 "$STORY" "$ODF"
echo "    Ornith DFlash2: $(tps of-32 of-160 128)"; grep -hE "spec:|spec time" "$OUT/of-160.log" | sed 's/^/   /'

echo "== Muse-Glimmer-30B INT4-W4A16, pure-mode batched prefill (was 475.6 s long)"
MU=Muse-Glimmer-30B-INT4-W4A16
run mu-s $MU int4 64 "$SHORT"; echo "    text: $(txt mu-s | tr '\n' ' ' | cut -c1-220)"
run mu-l $MU int4 24 "$LONG"; echo "    text: $(txt mu-l | tr '\n' ' ' | cut -c1-220)"
run mu-sq $MU int4 64 "$SHORT" "GRIMOIRE_MUSE_SEQUENTIAL_PREFILL=1"
echo "    short text batched vs sequential prefill: $(same mu-s mu-sq)"
echo "ALL DONE"
