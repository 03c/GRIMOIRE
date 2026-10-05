#!/bin/bash
# validate_image_v17.sh -- release image v1.7, gpu0 only: the v1.6 Ornith checks (single user,
# 8 slots at pp512 and pp4096) plus the Qwen3.8-27B recipes this release is about (INT4 GPTQ,
# MTP, DFlash2).  llama-benchy reads its book from the local copy (gutenberg.org can time out).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; PORT=6901
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl"
up() {  # up NAME MODEL PROJ ENV [SERVER ARGS]
  for i in 1 2 3; do
    SRV_MODE=image SRV_IMAGE=grimoire-b70:latest SRV_ENV="$4" SRV_ARGS="${5:-}" SRV_WAIT=1800 \
      bash $B/srv2.sh up $1 gpu0 $PORT $2 $3 | grep -E "scheduler|ready" && return 0
    sleep 5
  done
  return 1
}
bench() {  # bench NAME MODEL OUT ARGS...
  local nm=$1 m=$2 out=$3; shift 3
  timeout 2400 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$m \
    --tokenizer /mnt/storage/Models/$m $BOOK "$@" --format md --save-result $B/$out.md > $B/logs/$out.out 2>&1
  echo "benchy rc=$?"; grep -E "Coherence" $B/logs/$out.out; grep -E "^\|" $B/$out.md
  bash $B/srv2.sh down $nm $PORT
}
docker image inspect grimoire-b70:latest --format "image {{.Id}} created {{.Created}} size {{.Size}}"
M=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
echo "== Ornith default (one request at a time)"
up img-v17 $M mxfp4 "$BASE" || exit 1
curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/'$M'","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40}' | head -c 400; echo
bench img-v17 $M gap-image-v17 --pp 128 2048 4096 --tg 32 256 --runs 3
echo "== Ornith GRIMOIRE_SEQ_SLOTS=8, pp512/tg64"
up img-v17s $M mxfp4 "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
bench img-v17s $M conc-image-v17 --pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2
echo "== Ornith GRIMOIRE_SEQ_SLOTS=8, pp4096/tg32"
up img-v17l $M mxfp4 "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
bench img-v17l $M conc4k-image-v17 --pp 4096 --tg 32 --concurrency 1 2 4 8 --runs 2
Q=Qwen3.8-27B-FP8
echo "== $Q + DFlash2 (z-lab/Qwen3.8-27B-DFlash2)"
up img-v17qd $Q mxfp4 "$BASE" "--dflash-model /models/Qwen3.8-27B-DFlash2" || exit 1
bench img-v17qd $Q q38-dflash2-image-v17 --pp 512 2048 --tg 128 --runs 2
echo "== $Q + MTP k=3"
up img-v17qm $Q mxfp4 "$BASE
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3" || exit 1
bench img-v17qm $Q q38-mtp-image-v17 --pp 512 2048 --tg 128 --runs 2
S=Swift-1.5-Qwen3.8-27B-GPTQ-Int4-MTP-BF16
echo "== $S GRIMOIRE_SEQ_SLOTS=8"
up img-v17sw $S int4 "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
bench img-v17sw $S swift-image-v17 --pp 512 --tg 128 --concurrency 1 2 4 8 --runs 2
echo "== $S + MTP k=3"
up img-v17swm $S int4 "$BASE
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3" || exit 1
bench img-v17swm $S swift-mtp-image-v17 --pp 512 2048 --tg 128 --runs 2
echo "== $S + DFlash2"
up img-v17swd $S int4 "$BASE" "--dflash-model /models/Qwen3.8-27B-DFlash2" || exit 1
bench img-v17swd $S swift-dflash2-image-v17 --pp 512 2048 --tg 128 --runs 2
echo ALL DONE
