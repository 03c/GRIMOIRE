#!/bin/bash
# smallm_ab.sh TAG [ENV lines] -- Ornith MXFP4 server on gpu0 (GRIMOIRE_SEQ_SLOTS=8 + ENV),
# 8 concurrent greedy chats (texts saved), prof_batch wall M=1/2/4/8, then a graceful stop.
B=/mnt/storage/isos/grimoire-runs/bench-1003
TAG=$1; EXTRA=${2:-}
MODEL=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
mkdir -p $B/ab
export SRV_MODE=hostbin SRV_WAIT=1200
export SRV_ENV="GRIMOIRE_SEQ_SLOTS=8
$EXTRA"
bash $B/srv.sh up ab-$TAG gpu0 6990 $MODEL mxfp4 || exit 1
python3 $B/texts8.py 6990 $MODEL > $B/ab/$TAG.texts.json 2> $B/ab/$TAG.texts.err
python3 $B/prof_batch.py 6990 $MODEL ab-$TAG wall > $B/ab/$TAG.wall.txt 2>&1
bash $B/srv.sh down ab-$TAG 6990
cat $B/ab/$TAG.wall.txt
