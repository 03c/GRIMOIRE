#!/bin/bash
# vllm_conc.sh -- the REFERENCE: vLLM (its own container, the competitor) on gpu0 only,
# Ornith-1.5-35B-A3B-GPTQ-Int4, Ian's newest settings (my-VLLM-XPU-35B-NEW) minus MTP,
# max-num-seqs 8.  Same llama-benchy tests as the GRIMOIRE sweep.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; mkdir -p $B/vllm
M=Ornith-1.5-35B-A3B-GPTQ-Int4; PORT=1600; NAME=vllm-ref
cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh gpu0) || exit 2
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --ipc=host --shm-size=10g --network host --init --stop-timeout 300 \
  --device /dev/dri/$NODE -v /mnt/storage/Models:/models \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e ZE_AFFINITY_MASK=0 -v /mnt/storage/isos/grimoire-runs/bench-1003/vllm/qwen3_dflash.resolved.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/intel/oneapi/ccl/latest/lib:/opt/intel/oneapi/mpi/latest/lib:/opt/intel/oneapi/tcm/1.5/lib:/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:/usr/local/lib \
  my-vllm-xpu:latest /models/$M --dtype float16 --port $PORT --host 0.0.0.0 --max-model-len 8192 \
  --max-num-seqs 8 --kv-cache-dtype fp8 --max-num-batched-tokens 16384 --block-size 64 \
  --language-model-only --trust-remote-code --gpu-memory-utilization 0.90 >/dev/null || exit 3
for i in $(seq 1 200); do
  sleep 6
  curl -s -m 3 http://localhost:$PORT/v1/models | grep -q '"id"' && { echo "vLLM ready after $((i*6))s"; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "vLLM EXITED:"; docker logs $NAME 2>&1 | tail -25; exit 4; }
done
docker logs $NAME 2>&1 | grep -iE "graph|backend|kv cache|maximum concurrency|Using|quantiz" | grep -viE "warn.*deprecat" | head -15 | cut -c1-200
ID=$(curl -s http://localhost:$PORT/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
timeout 2400 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model "$ID" --tokenizer /mnt/storage/Models/$M \
  --pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2 --format md --save-result $B/vllm/conc.md > $B/vllm/conc.log 2>&1
echo "conc rc=$?"; grep -E "Coherence" $B/vllm/conc.log; grep -E "^\|" $B/vllm/conc.md
timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model "$ID" --tokenizer /mnt/storage/Models/$M \
  --pp 128 2048 4096 --tg 256 --runs 2 --format md --save-result $B/vllm/single.md > $B/vllm/single.log 2>&1
echo "single rc=$?"; grep -E "^\|" $B/vllm/single.md
for i in $(seq 1 60); do [ "$(ss -Htn state established "( sport = :$PORT )" | wc -l)" -eq 0 ] && break; sleep 2; done
docker logs $NAME > $B/vllm/server.log 2>&1; docker stop --time 120 $NAME >/dev/null; docker rm $NAME >/dev/null
echo ALL DONE
