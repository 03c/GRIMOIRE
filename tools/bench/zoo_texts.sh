#!/bin/bash
# zoo_texts.sh MODEL PROJ [ENV lines] -- batched (8 concurrent) greedy texts on gpu0, for coherence
B=/mnt/storage/isos/grimoire-runs/bench-1003; M=$1; P=$2; EXTRA=${3:-}
export SRV_MODE=hostbin SRV_WAIT=1500
export SRV_ENV="GRIMOIRE_SEQ_SLOTS=8
$EXTRA"
n=zt-$(echo $M | tr "/." "__" | cut -c1-40)
for i in 1 2 3; do bash $B/srv.sh up $n gpu0 6990 $M $P >/dev/null && break; sleep 5; done || { echo "UP FAILED $M"; exit 1; }
timeout 900 python3 $B/texts8.py 6990 $M 40 > $B/ab/zt-$(basename $M).json 2>&1 || echo "TEXTS FAILED"
bash $B/srv.sh down $n 6990 >/dev/null
echo "== $M"; python3 -c "import json,sys; d=json.load(open(\"$B/ab/zt-$(basename $M).json\")); [print(k, repr(v[:110])) for k,v in d.items()]" 2>&1 | head -9
