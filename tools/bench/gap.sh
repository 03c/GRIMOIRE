#!/bin/bash
# gap.sh -- why llama-benchy TG (101-128) != CLI TG (198.6) on Ornith.  gpu0 only.
# Same first request as yesterday (a short smoke chat, which is what captures the
# decode graph), then the same benchy, under three configurations.
set -u
B="${BENCH_OUT:-/mnt/storage/isos/grimoire-runs/bench}"; mkdir -p $B/logs
HERE="$(cd "$(dirname "$0")" && pwd)"
M=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE; PORT=6901
bench() {
  timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
    --tokenizer /mnt/storage/Models/$M --pp 128 2048 4096 --tg 32 256 --runs 3 \
    --format md --save-result $B/gap-$1.md > $B/logs/gap-$1.out 2>&1
  echo "== $1 (rc=$?)"; grep -E "^\|" $B/gap-$1.md 2>/dev/null || tail -5 $B/logs/gap-$1.out
}
smoke() { curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/'$M'","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' | head -c 400; echo; }
run_cfg() { # tag mode image env
  SRV_MODE=$2 SRV_IMAGE=$3 SRV_ENV="$4" bash $HERE/srv.sh up g-$1 gpu0 $PORT $M mxfp4 || { echo "server $1 failed"; return 1; }
  smoke; bench $1; bash $HERE/srv.sh down g-$1 $PORT
}
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl"
run_cfg b70-v2off     image grimoire-b70:latest "$BASE
SYCL_UR_USE_LEVEL_ZERO_V2=0"
run_cfg b70-v2on      image grimoire-b70:latest "$BASE"
run_cfg b70-v2on-s128 image grimoire-b70:latest "$BASE
GRIMOIRE_DECODE_SPLITS=128"
echo ALL DONE
