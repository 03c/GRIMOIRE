**Concurrent serving: 8 users at 255 tok/s total (v1.2: 174). Single-user speed unchanged.**

Measured with llama-benchy 0.4.0 on one Arc Pro B70, Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE, `GRIMOIRE_SEQ_SLOTS=8`, pp512/tg64. Totals are decode tok/s; per-request values are in parentheses.

| concurrent requests | v1.2 | **v1.3** | peak |
|---|---:|---:|---:|
| 1 | 193 | **193.4** | 196 |
| 2 | 95 | **145.8** (79) | 148 |
| 4 | 133 | **198.3** (58) | 230 |
| 8 | 174 | **255.2** (41) | 332 |

Single request, default server (no `GRIMOIRE_SEQ_SLOTS`), unchanged:

| context | prefill | decode (tg256) |
|---|---:|---:|
| ~0 tokens | ~4,000 tok/s | **195.5 tok/s** |
| 2K tokens | 9,550 tok/s | **183.3 tok/s** |
| 4K tokens | 9,915 tok/s | **172.0 tok/s** |

### What changed
Batched decode now runs on GRIMOIRE's own kernels, written for this case. With N requests in flight, one decode step is one M=N batch.

- **Small-batch GEMMs on the matrix engines.** Every projection, the lm_head, the router and the DeltaNet gates read their weights once per step for all rows. MXFP4 weights are decoded exactly on the ALU, and BF16 weights need no decode at all. ([4706273](https://github.com/doopeworld/GRIMOIRE/commit/4706273), [0dc218b](https://github.com/doopeworld/GRIMOIRE/commit/0dc218b))
- **Grouped MoE.** Each touched expert is streamed once for all of its routed rows. Gate_up uses a layout with 64-byte weight rows. ([4706273](https://github.com/doopeworld/GRIMOIRE/commit/4706273), [0dc218b](https://github.com/doopeworld/GRIMOIRE/commit/0dc218b))
- **One launch per layer for all rows.** This covers the DeltaNet state update, the causal conv, rope, KV append and flash-decode attention. Each row still uses its own conversation's state, cache, length and split count, and is bit-identical to the per-row path. ([4706273](https://github.com/doopeworld/GRIMOIRE/commit/4706273), [18d07bc](https://github.com/doopeworld/GRIMOIRE/commit/18d07bc))

GPU time for one 8-row Ornith decode step fell from ~37 ms to **19.2 ms**. The scheduler adds nothing measurable.

These kernels are used only for batched decode across conversations. Single-request serving and speculative decoding run exactly as in v1.2. To turn the new kernels off: `GRIMOIRE_SMALLM_DPAS=0`, `GRIMOIRE_MOE_SMALLM_DPAS=0`, `GRIMOIRE_ROWS_BATCH=0`.

### Still open
New prompts are admitted one at a time: each prompt's prefill runs alone while the other requests wait. That costs about a quarter of the c8 run above (pp512 at c8: 4,560 tok/s total). Batching admission is next.

### Use
```
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.3/grimoire-b70-v1.3.tar.gz
docker load < grimoire-b70-v1.3.tar.gz          # -> grimoire-b70:latest
docker run -d --name grimoire --network host --init --device /dev/dri/renderD128 \
  -e GRIMOIRE_SEQ_SLOTS=8 \
  -v /path/to/models:/models grimoire-b70:latest \
  server --model /models/<checkpoint> --proj mxfp4 --ctx 8192 --port 8000
```
`GRIMOIRE_SEQ_SLOTS=8` serves up to 8 requests at once. Leave it out to serve one request at a time.

Pick the render node of your B70 with `ls -l /dev/dri/by-path/`. Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

sha256 `34a1483c8941b03401160f7160f45b94a0d267f7d02567b813f77867c82668f5`
