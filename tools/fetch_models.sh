#!/bin/bash
# fetch_models.sh -- download the checkpoints GRIMOIRE supports but never ran on
# the B70 (earlier sessions had no HuggingFace access).  Runs the HuggingFace CLI
# on the host through uv; no container, no GPU.  Log: /mnt/storage/Models/_downloads.log
set -u
LOG=/mnt/storage/Models/_downloads.log
for r in unsloth/Qwen3.8-27B-NVFP4 ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
         IFM/K2-Horizon-MoVA-36B-A4B Agnes-AI/Agnes-3.0-Flash; do
  d=/mnt/storage/Models/$(basename $r)
  echo "$(date +%T) START $r -> $d" >> $LOG
  uvx --from huggingface_hub hf download "$r" --local-dir "$d" --max-workers 8 > /dev/null 2>&1
  echo "$(date +%T) DONE  $r exit $? ($(du -sh "$d" | cut -f1))" >> $LOG
done
echo "$(date +%T) ALL DONE" >> $LOG
