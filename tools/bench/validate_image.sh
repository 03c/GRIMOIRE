#!/bin/bash
# validate_image.sh -- the release image, run like GRIMOIRE-ORNITH (minus the legacy-adapter
# override), short first request, then the same llama-benchy matrix. gpu0 only.
set -u
B="${BENCH_OUT:-/mnt/storage/isos/grimoire-runs/bench}"; mkdir -p $B/logs
HERE="$(cd "$(dirname "$0")" && pwd)"
M=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE; PORT=6901
docker image inspect grimoire-b70:latest --format "image {{.Id}} created {{.Created}} size {{.Size}}"
SRV_MODE=image SRV_IMAGE=grimoire-b70:latest SRV_ENV="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl" bash $HERE/srv.sh up img-v11 gpu0 $PORT $M mxfp4 || exit 1
docker logs img-v11 2>&1 | grep -iE "level.zero|adapter|V2" | head -3
curl -s -m 60 http://localhost:$PORT/v1/models; echo
curl -s -m 120 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/'$M'","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40}' | head -c 600; echo
timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
  --tokenizer /mnt/storage/Models/$M --pp 128 2048 4096 --tg 32 256 --runs 3 \
  --format md --save-result $B/gap-image-v11.md > $B/logs/gap-image-v11.out 2>&1
echo "benchy rc=$?"; grep -E "Coherence" $B/logs/gap-image-v11.out; grep -E "^\|" $B/gap-image-v11.md
bash $HERE/srv.sh down img-v11 $PORT
echo ALL DONE
