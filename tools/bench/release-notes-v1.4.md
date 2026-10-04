**Concurrent serving, round 2: c8 348–392 tok/s total (v1.3: 255). Every concurrency level now scales above single-user speed. Single-user speed is unchanged.**

Measured with llama-benchy 0.4.0 on one Arc Pro B70, Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE, `GRIMOIRE_SEQ_SLOTS=8`, pp512/tg64. Totals are decode tok/s; per-request values are in parentheses.

| concurrent requests | v1.2 | v1.3 | **v1.4** | v1.4 peak |
|---|---:|---:|---:|---:|
| 1 | 193 | 193.4 | **193.5** | 197 |
| 2 | 95 | 145.8 | **215.0** (107) | 218 |
| 4 | 133 | 198.3 | **264–293** (77–80) | 298 |
| 8 | 174 | 255.2 | **348–392** (51–53) | 414–423 |

Concurrent prompt processing (pp512, 8 at once): **7,980–8,020 tok/s** total. v1.3 managed 4,560.

Single request on the default server (no `GRIMOIRE_SEQ_SLOTS`) is unchanged:

| context | prefill | decode (tg256) |
|---|---:|---:|
| ~0 tokens | ~4,100 tok/s | **195.6 tok/s** |
| 2K tokens | 9,530 tok/s | **183.5 tok/s** |
| 4K tokens | 9,910 tok/s | **172.3 tok/s** |

The c4 and c8 totals vary between runs. They depend on how simultaneous requests group into admission batches.

### What changed
- **Batched admission** ([e449b7c](https://github.com/doopeworld/GRIMOIRE/commit/e449b7c)). Prompts that arrive together are prefilled in one forward pass instead of one by one.
  - Per prompt: each runs its own conv, DeltaNet recurrence and causal attention.
  - Shared work: projections, MoE and norms read their weights once for all of them.
  - A single 512-token prompt touches every expert of the model, so one-at-a-time admission read all of the weights again for each prompt.
  - Turn it off with `GRIMOIRE_ADMIT_BATCH_TOKENS=0`. The default batch cap is 8192 tokens.
- **Decode MoE over all rows** ([bd8b066](https://github.com/doopeworld/GRIMOIRE/commit/bd8b066)). Batched decode runs the single-user decode step's MoE kernels, shared expert fused, over every row in two launches.
  - Each row's MoE output is bit-identical to a single-user decode step.
  - GPU time for an 8-row step: 19.3 → 17.0 ms. For a 2-row step: 11.2 → 8.7 ms.
  - Turn it off with `GRIMOIRE_MOE_ROWS_MAX=0`.

These paths serve batched decode and batched admission only. Single-request serving, speculative decoding and the prefix cache work exactly as in v1.3.

### Use
```
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.4/grimoire-b70-v1.4.tar.gz
docker load < grimoire-b70-v1.4.tar.gz          # -> grimoire-b70:latest
docker run -d --name grimoire --network host --init --device /dev/dri/renderD128 \
  -e GRIMOIRE_SEQ_SLOTS=8 \
  -v /path/to/models:/models grimoire-b70:latest \
  server --model /models/<checkpoint> --proj mxfp4 --ctx 8192 --port 8000
```
`GRIMOIRE_SEQ_SLOTS=8` serves up to 8 requests at once.

Find your B70's render node with `ls -l /dev/dri/by-path/`. The numbers can change after a reboot. Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

sha256 `6ed70d2f6185d6c8ac7a6131e3d017c624fa3221237a66146a9df1a11887abfd`
