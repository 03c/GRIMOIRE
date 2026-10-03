#!/bin/bash
# smallm_regions.sh TAG [ENV lines] -- like smallm_ab.sh but GRIMOIRE_TIME_LAYER=all + prof_batch regions
B=/mnt/storage/isos/grimoire-runs/bench-1003
TAG=$1; EXTRA=${2:-}
MODEL=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
export SRV_MODE=hostbin SRV_WAIT=1200
export SRV_ENV="GRIMOIRE_SEQ_SLOTS=8
GRIMOIRE_SCHED_SOLO=0
GRIMOIRE_TIME_LAYER=all
$EXTRA"
bash $B/srv.sh up rg-$TAG gpu0 6990 $MODEL mxfp4 >/dev/null || { echo "up failed"; exit 1; }
python3 $B/prof_batch.py 6990 $MODEL rg-$TAG regions > $B/ab/$TAG.regions.txt 2>&1
bash $B/srv.sh down rg-$TAG 6990 >/dev/null
cat $B/ab/$TAG.regions.txt
