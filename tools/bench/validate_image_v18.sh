#!/bin/bash
# validate_image_v18.sh -- release image v1.8, gpu0 only, llama-benchy 0.4.0 against the image's
# own grimoire-server: the Qwen3.8-27B GPTQ-Int4 + MTP recipe this release is about (new defaults:
# GRIMOIRE_MTP=1 alone = 4 drafts, MXFP4 draft head, drafting while <= 4 requests), 1-8 users at
# pp512 and pp4096, plus Ornith (single user and 8 slots) so the MXFP4 path is checked as well.
# llama-benchy reads its book from the local copy (gutenberg.org can time out).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; PORT=6901
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl"
up() {  # up NAME MODEL PROJ ENV [CTX]
  for i in 1 2 3; do
    SRV_MODE=image SRV_IMAGE=grimoire-b70:latest SRV_ENV="$4" SRV_CTX="${5:-8192}" SRV_WAIT=1800 \
      BENCH_OUT=$B bash $B/srv2.sh up $1 gpu0 $PORT $2 $3 | grep -E "scheduler|ready" && return 0
    sleep 5
  done
  return 1
}
bench() {  # bench NAME MODEL OUT ARGS...
  local nm=$1 m=$2 out=$3; shift 3
  timeout 3000 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$m \
    --tokenizer /mnt/storage/Models/$m $BOOK "$@" --format md --save-result $B/$out.md > $B/logs/$out.out 2>&1
  echo "benchy rc=$?"; grep -E "Coherence" $B/logs/$out.out; grep -E "^\|" $B/$out.md
  BENCH_OUT=$B bash $B/srv2.sh down $nm $PORT
}
mkdir -p $B/logs
docker image inspect grimoire-b70:latest --format "image {{.Id}} created {{.Created}} size {{.Size}}"
Q=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
echo "== $Q + MTP (defaults), GRIMOIRE_SEQ_SLOTS=8, --ctx 16384"
up img-v18q $Q int4 "$BASE
GRIMOIRE_MTP=1
GRIMOIRE_SEQ_SLOTS=8" 16384 || exit 1
curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/'$Q'","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40}' | head -c 400; echo
bench img-v18q $Q q38-gptq-mtp-image-v18 --pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2
O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
echo "== $O, one request at a time"
up img-v18o $O mxfp4 "$BASE" || exit 1
bench img-v18o $O ornith-image-v18 --pp 512 4096 --tg 128 --runs 2
echo "== $O, GRIMOIRE_SEQ_SLOTS=8, pp512/tg64"
up img-v18os $O mxfp4 "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
bench img-v18os $O ornith-conc-image-v18 --pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2
echo ALL DONE
