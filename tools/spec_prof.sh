#!/bin/bash
# spec_prof.sh -- where the DFlash draft second and the verify pass go. gpu0.
set -u
cd /mnt/storage/isos/grimoire-fuse
STORY="Write a detailed story about a lighthouse keeper."
r=$(EXTRA_ENV="GRIMOIRE_DFLASH_MODEL=/models/Qwen3.8-27B-DFlash2
GRIMOIRE_DFLASH_TIME=1
GRIMOIRE_SPEC_TIME=1" LIM=900 bash tools/g0run.sh sp-df -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192 -p "$STORY" -n 24) || { echo STOP; exit 9; }
echo "dflash: $r"
grep -iE "DFlash (draft|context)|draft call|ms  |ms$" /tmp/grim-sp-df.log | grep -v unavailable | head -60
r=$(EXTRA_ENV="GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_TIME_LAYER=all" LIM=900 bash tools/g0run.sh sp-mtp -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 --proj mxfp4 --ctx 8192 -p "$STORY" -n 12) || { echo STOP; exit 9; }
echo "mtp: $r"
grep -nE "host region budget|TOTAL|^\s+[a-zA-Z].*ms +[0-9.]+%" /tmp/grim-sp-mtp.log | tail -40
