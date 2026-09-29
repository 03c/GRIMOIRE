#!/bin/bash
# post_sweep.sh -- verify the GQA flash-decode kernel, the DFlash small-M
# drafter routing and the spec phase timer.  gpu0 only, via g0run.sh.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/post-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
STORY="Write a detailed story about a lighthouse keeper."
LONG="$(cat real4k_ascii.txt)"
run() { local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh ps-$nm -m /models/$m --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP at $nm: $r"; cp /tmp/grim-ps-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-ps-$nm.log "$OUT/$nm.log"; echo "  $nm: ${r:-NO RESULT LINE}"; }
txt() { awk '/norms /{f=1;next} f' "$OUT/$1.log" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$OUT/$1.log" | tail -1 | cut -d= -f2; }
tps() { python3 -c "a=float('$(el $1)');b=float('$(el $2)');per=(b-a)/$3;print(f'{per*1000:.2f} ms/token = {1/per:.1f} tok/s')"; }
same() { cmp -s <(txt $1) <(txt $2) && echo IDENTICAL || echo differs; }
firstdiff() { python3 - "$OUT/$1.log" "$OUT/$2.log" <<'PY'
import sys,re
def t(p):
    s=open(p,errors='ignore').read().split('\n'); o=[];f=False
    for l in s:
        if re.search(r'norms ',l): f=True; continue
        if f and l and not l[0].isspace() and 'prompt=' not in l and 'ids' not in l: o.append(l)
    return '\n'.join(o)
a,b=t(sys.argv[1]),t(sys.argv[2]); n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
print(f"    first difference at char {n} of {len(a)}/{len(b)}: {a[max(0,n-40):n+30]!r} | {b[max(0,n-40):n+30]!r}")
PY
}

K2=K2-Horizon-MoVA-36B-A4B
echo "== flash_decode_gqa on K2 (old kernel = GRIMOIRE_FLASH_DECODE_OLD=1)"
run k-new $K2 mxfp4 256 "$SHORT"
run k-old $K2 mxfp4 256 "$SHORT" "GRIMOIRE_FLASH_DECODE_OLD=1"
echo "    256-token text new vs old: $(same k-new k-old)"; [ "$(same k-new k-old)" = differs ] && firstdiff k-new k-old
run k-tl $K2 mxfp4 8 "$LONG" "GRIMOIRE_TIMELINE=1
GRIMOIRE_TIMELINE_LAYER=3"
grep -E "flash_decode|flash_merge|  TOTAL" "$OUT/k-tl.log" | head -3 | sed 's/^/   /'
run k-ln $K2 mxfp4 64 "$LONG"
run k-lo $K2 mxfp4 64 "$LONG" "GRIMOIRE_FLASH_DECODE_OLD=1"
echo "    5.7K context, 64 tokens: new $(el k-ln) s  old $(el k-lo) s  text: $(same k-ln k-lo)"

Q=Qwen3.8-27B-MXFP4-GRIMOIRE
echo "== Qwen3.8-27B with the new decode attention"
run q-c1 $Q mxfp4 24 "$LONG"
cmp -s <(txt q-c1) ref-sherlock-5987-n24.txt && echo "    Sherlock n=24: IDENTICAL to reference" || { echo "    Sherlock n=24: DIFFERS"; diff <(txt q-c1) ref-sherlock-5987-n24.txt | head -4; }
run q-l64 $Q mxfp4 64 "$LONG"
run q-l64o $Q mxfp4 64 "$LONG" "GRIMOIRE_FLASH_DECODE_OLD=1"
echo "    5987 ctx, 64 tokens: new $(el q-l64) s  old $(el q-l64o) s  text: $(same q-l64 q-l64o)"

echo "== DFlash2 drafter small-M routing + phase timer"
DF="GRIMOIRE_DFLASH_MODEL=/models/Qwen3.8-27B-DFlash2
GRIMOIRE_SPEC_TIME=1"
run qf-32 $Q mxfp4 32 "$STORY" "$DF"
run qf-160 $Q mxfp4 160 "$STORY" "$DF"
echo "    Qwen DFlash2: $(tps qf-32 qf-160 128)"; grep -hE "spec:|spec time" "$OUT/qf-160.log" | sed 's/^/   /'
O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
ODF="GRIMOIRE_DFLASH_MODEL=/models/Ornith-1.5-35B-A3B-DFlash2
GRIMOIRE_SPEC_TIME=1"
run of-32 $O mxfp4 32 "$STORY" "$ODF"
run of-160 $O mxfp4 160 "$STORY" "$ODF"
echo "    Ornith DFlash2 (mxfp4 drafter): $(tps of-32 of-160 128)"; grep -hE "spec:|spec time" "$OUT/of-160.log" | sed 's/^/   /'
run ob-160 $O mxfp4 160 "$STORY" "$ODF
GRIMOIRE_DFLASH_DRAFT_BF16=1"
echo "    Ornith DFlash2 (bf16 drafter):"; grep -hE "spec:|spec time" "$OUT/ob-160.log" | sed 's/^/   /'
QM=Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16
run qm-160 $QM mxfp4 160 "$STORY" "GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_SPEC_TIME=1"
echo "    Qwen MTP:"; grep -hE "spec:|spec time" "$OUT/qm-160.log" | sed 's/^/   /'
echo "ALL DONE"
