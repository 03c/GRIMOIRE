#!/bin/bash
# vllm_mtp_benchy.sh -- llama-benchy 0.4.0 against the vLLM competitor container (:8100), with MTP counters.
B=/mnt/storage/isos/grimoire-runs/bench-1003; Z=$B/vllm-cmp; mkdir -p $Z
M=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
cnt() { curl -s http://127.0.0.1:8100/metrics | awk '/^vllm:spec_decode_num_(accepted_tokens|draft_tokens|drafts)_total/{split($1,a,"{"); s[a[1]]+=$2} END{printf "%d %d %d", s["vllm:spec_decode_num_accepted_tokens_total"], s["vllm:spec_decode_num_draft_tokens_total"], s["vllm:spec_decode_num_drafts_total"]}'; }
for C in "1" "2 4 8"; do
  tag=c$(echo $C | tr -d ' ')
  read a0 d0 n0 <<< "$(cnt)"
  timeout 3000 uvx llama-benchy@0.4.0 --base-url http://localhost:8100/v1 --model qwen38 --tokenizer /mnt/storage/Models/$M \
    --book-url http://127.0.0.1:8999/1661-0.txt --pp 512 4096 --tg 128 --concurrency $C --runs 2 --format md \
    --save-result $Z/benchy-$tag.md > $Z/benchy-$tag.log 2>&1
  read a1 d1 n1 <<< "$(cnt)"
  echo "== $tag rc=$? MTP accepted $((a1-a0))/$((d1-d0)) drafted, $(python3 -c "print(f'{($a1-$a0)/max($d1-$d0,1)*100:.1f}% , {($a1-$a0)/max($n1-$n0,1)+1:.2f} tokens/step')")"
  grep -E "^\|" $Z/benchy-$tag.md
done
echo VLLM BENCHY DONE
