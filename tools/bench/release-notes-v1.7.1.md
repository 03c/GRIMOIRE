## GRIMOIRE v1.7.1 -- several users on Qwen3.8-27B (with or without MTP), no VRAM overflow

```
docker load < grimoire-b70-v1.7.1.tar.gz      # -> grimoire-b70:latest
```

### Fixes

**Prefill no longer overflows VRAM on big dense models.** A prompt's prefill scratch is about
1 MB per token on Qwen3.8-27B (its feed-forward is 34,816 wide). Eight concurrent 4K prompts
were admitted as one 32K-token batch, which asked for ~30 GB: the driver started moving memory
to system RAM, decode steps took ~10 s and the GPU could reset. v1.7.1 sizes every prefill
from the card's free VRAM (printed at startup as `prefill chunk up to N tokens per call`):
bursts are admitted in waves that fit, and a prompt longer than that is processed in chunks.
`GRIMOIRE_PREFILL_CHUNK=<tokens>` overrides it.

**MTP together with `GRIMOIRE_SEQ_SLOTS`.** With several requests in flight, MTP now drafts only
while at most 2 sequences are active (`GRIMOIRE_SPEC_MAX_SEQS`, default 2) and switches to plain
batched decoding above that; prompts are admitted in batches with MTP on too.

Qwen3.8-27B GPTQ-Int4 ([SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)),
`-e GRIMOIRE_MTP=1 -e GRIMOIRE_MTP_K=3 -e GRIMOIRE_SEQ_SLOTS=8 --ctx 16384`, llama-benchy 0.4.0,
total generated tokens/s at 1 / 2 / 4 / 8 users:

| Test | v1.7 | v1.7.1 |
|---|---|---|
| pp512 / tg128 | 53.8 / 58.5 / 61.7 / 76.2 | 49.0 / 67.4 / 84.2 / **139.2** |
| pp4096 / tg128 | 50.7 / 38.8 / 29.5 / 32.5 | 51.1 / 38.4 / 36.2 / 39.7 |

Single-user MTP is the same within run-to-run noise (3-run A/B: 52.8 +- 2.7 vs 49.7 +- 0.7).
Ornith-1.5-35B-A3B (8 slots, `--ctx 32000`), pp4096/tg32: 171.2 / 213.6 / 299.4 / 393.5
(v1.6: 171 / 212 / 296 / 390).

### Settings advice

- **Context with 8 slots:** the KV cache is reserved for every slot. On Qwen3.8-27B,
  `--ctx 32000` x 8 slots takes 8.4 GB and leaves room for only ~1,500 prompt tokens per
  prefill call; `--ctx 16384` x 8 slots leaves ~5,900. Use the smallest context you need.
- **Long prompts with many users** on Qwen3.8-27B are still admitted one after another
  (one 4K prompt fills a prefill call); faster INT4 prefill and smaller prefill scratch are next.

### Image

`grimoire-b70-v1.7.1.tar.gz`, 640,771,417 bytes, sha256
`d3061a2c820889ac31656a3c1cd295bb2b8900ff13f5e882f14f8747e0300df3`, built from commit 9c403c8
(image id `22169e249ef9`).
