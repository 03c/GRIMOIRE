#!/bin/bash
# vllm_mtp_up.sh -- COMPETITOR measurement only: vLLM in its own image and container (nothing
# shared with GRIMOIRE), gpu0 only, with the B70 inference cookbook's flags for
# SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 (MTP depth ${K:-4}, fp8 KV, GPTQ), no cookbook
# patches.  Used for the v1.8 release comparison (2026-10-06): llama-benchy via vllm_mtp_benchy.sh,
# same prompts as GRIMOIRE via spec_probe.py.  The local my-vllm-xpu image has one unresolved
# merge-conflict hunk in qwen3_dflash.py; vllm_mtp_fix.py resolves it inside the container.
# Stop it when idle: docker stop --time 120 zc-vllm-mtp && docker rm zc-vllm-mtp
T=$(cd "$(dirname "$0")" && pwd)
NODE=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)
[ "$NODE" = /dev/dri/renderD128 ] || { echo "gpu0 is $NODE, expected renderD128 -- refusing"; exit 9; }
docker rm -f zc-vllm-mtp >/dev/null 2>&1
docker run -d --name zc-vllm-mtp --init --stop-timeout 300 \
  --device /dev/dri/renderD128 --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v /mnt/storage/Models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16:/model:ro -p 8100:8000 \
  -v $T/vllm_mtp_fix.py:/fix.py:ro -v $T/vllm_mtp_run.sh:/run.sh:ro \
  -e K=${K:-4} -e ZE_AFFINITY_MASK=0 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash my-vllm-xpu:latest /run.sh
echo started
