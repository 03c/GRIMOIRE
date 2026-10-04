#!/bin/bash
# grim_conc.sh TAG [ENV lines] -- GRIMOIRE Ornith MXFP4 on gpu0, GRIMOIRE_SEQ_SLOTS=8 + ENV,
# llama-benchy pp512/tg64 c1 2 4 8 (the vLLM reference settings), graceful stop.
B=/mnt/storage/isos/grimoire-runs/bench-1003; mkdir -p $B/conc
TAG=$1; EXTRA=${2:-}; M=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE; PORT=6990
export SRV_MODE=hostbin SRV_WAIT=1200
export SRV_ENV="GRIMOIRE_SEQ_SLOTS=8
$EXTRA"
for i in 1 2 3; do bash $B/srv.sh up cc-$TAG gpu0 $PORT $M mxfp4 >/dev/null && break; sleep 5; done
timeout 2400 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M --tokenizer /mnt/storage/Models/$M \
  --pp ${PP:-512} --tg ${TG:-64} --concurrency 1 2 4 8 --runs ${RUNS:-2} --format md --save-result $B/conc/$TAG.md > $B/conc/$TAG.log 2>&1
echo "rc=$?"; grep -E "Coherence" $B/conc/$TAG.log | head -3; grep -E "^\|" $B/conc/$TAG.md
bash $B/srv.sh down cc-$TAG $PORT >/dev/null
