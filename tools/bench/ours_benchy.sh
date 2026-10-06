#!/bin/bash
# ours_benchy.sh LIST -- llama-benchy 0.4.0 on grimoire-server built from the working tree (hostbin), gpu0.
# LIST lines: TAG|ENV (K=V,K=V)|CONC|PP list
B=/mnt/storage/isos/grimoire-runs/bench-1003; Z=$B/${ZDIR:-ours}; mkdir -p $Z
M=Qwen3.8-27B-GPTQ-Int4-MTP-BF16; PORT=6990
while IFS='|' read -r TAG EXTRA CONC PPS; do
  [ -z "$TAG" ] && continue; case "$TAG" in \#*) continue;; esac
  export SRV_MODE=${SRV_MODE_OVERRIDE:-hostbin} SRV_WAIT=1800 SRV_CTX=16384 BENCH_OUT=$B
  export SRV_ENV="$(echo "GRIMOIRE_SPEC_STATS=1,GRIMOIRE_SEQ_SLOTS=8,$EXTRA" | tr "," "\n")"
  bash $B/srv2.sh up zc-$TAG gpu0 $PORT $M int4 > $Z/$TAG.up 2>&1 || { echo "$TAG: UP FAILED: $(tail -3 $Z/$TAG.up | tr '\n' ' ')"; continue; }
  timeout 3000 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M --tokenizer /mnt/storage/Models/$M \
    --book-url http://127.0.0.1:8999/1661-0.txt --pp ${PPS:-512 4096} --tg 128 --concurrency $CONC --runs ${RUNS:-2} --format md --save-result $Z/$TAG.md > $Z/$TAG.log 2>&1
  rc=$?
  sleep 2
  docker logs zc-$TAG 2>&1 | grep "spec:" | python3 -c "
import sys,re
a=d=s=0
for l in sys.stdin:
    m=re.search(r'\((\d+) of (\d+) drafted.*over (\d+) steps',l)
    if m: a+=int(m[1]); d+=int(m[2]); s+=int(m[3])
print(f'MTP accepted {a}/{d} ({100*a/max(d,1):.1f}%), {a/max(s,1)+1:.2f} tokens/step' if d else 'no spec lines')" > $Z/$TAG.spec
  bash $B/srv2.sh down zc-$TAG $PORT >/dev/null
  echo "== $TAG rc=$rc coherent=$(grep -c 'Coherence test PASSED' $Z/$TAG.log)  $(cat $Z/$TAG.spec)"
  grep -E "^\| /models" $Z/$TAG.md | awk -F'|' '{printf "   %-14s %s\n", $3, $4}'
done < "$1"
echo OURS BENCHY DONE
