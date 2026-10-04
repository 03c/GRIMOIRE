#!/bin/bash
# validate_image_v16.sh -- release image: default config + GRIMOIRE_SEQ_SLOTS=8. gpu0 only.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003
M=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE; PORT=6901
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl"
up() {  # up NAME ENV
  for i in 1 2 3; do
    SRV_MODE=image SRV_IMAGE=grimoire-b70:latest SRV_ENV="$2" bash $B/srv.sh up $1 gpu0 $PORT $M mxfp4 | grep scheduler && return 0
    sleep 5
  done
  return 1
}
docker image inspect grimoire-b70:latest --format "image {{.Id}} created {{.Created}} size {{.Size}}"
echo "== default (one request at a time)"
up img-v16 "$BASE" || exit 1
curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/'$M'","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40}' | head -c 400; echo
timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
  --tokenizer /mnt/storage/Models/$M --pp 128 2048 4096 --tg 32 256 --runs 3 \
  --format md --save-result $B/gap-image-v16.md > $B/logs/gap-image-v16.out 2>&1
echo "benchy rc=$?"; grep -E "Coherence" $B/logs/gap-image-v16.out; grep -E "^\|" $B/gap-image-v16.md
bash $B/srv.sh down img-v16 $PORT
echo "== GRIMOIRE_SEQ_SLOTS=8 (batching + solo fast path)"
up img-v16s "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
  --tokenizer /mnt/storage/Models/$M --pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2 \
  --format md --save-result $B/conc-image-v16.md > $B/logs/conc-image-v16.out 2>&1
echo "benchy rc=$?"; grep -E "Coherence" $B/logs/conc-image-v16.out; grep -E "^\|" $B/conc-image-v16.md
bash $B/srv.sh down img-v16s $PORT
echo ALL DONE
echo "== GRIMOIRE_SEQ_SLOTS=8, pp4096/tg32 (long prompts)"
up img-v16l "$BASE
GRIMOIRE_SEQ_SLOTS=8" || exit 1
timeout 2400 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
  --tokenizer /mnt/storage/Models/$M --pp 4096 --tg 32 --concurrency 1 2 4 8 --runs 2 \
  --format md --save-result $B/conc4k-image-v16.md > $B/logs/conc4k-image-v16.out 2>&1
echo "benchy rc=$?"; grep -E "Coherence" $B/logs/conc4k-image-v16.out; grep -E "^\|" $B/conc4k-image-v16.md
bash $B/srv.sh down img-v16l $PORT
echo ALL4K DONE
