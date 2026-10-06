#!/bin/bash
# vllm_mtp_run.sh -- entrypoint of the vllm_mtp_up.sh competitor container.
python3 /fix.py || exit 1
exec vllm serve /model --quantization gptq --dtype float16 \
  --max-model-len 16384 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --no-enable-prefix-caching \
  --served-model-name qwen38 --language-model-only \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K:-4}}"
