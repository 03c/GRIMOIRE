**Concurrent serving, round 3: 8 users at 464 tok/s total, 4 users at 355, 2 at 238. Results are now consistent from run to run. Single-user speed is unchanged.**

Measured with llama-benchy 0.4.0 on one Arc Pro B70, Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE, `GRIMOIRE_SEQ_SLOTS=8`, pp512/tg64. Totals are decode tok/s; per-request values are in parentheses.

| concurrent requests | v1.3 | v1.4 | **v1.5** | v1.5 peak |
|---|---:|---:|---:|---:|
| 1 | 193.4 | 193.5 | **193.6** | 197 |
| 2 | 145.8 | 215.0 | **237.8** (119) | 242 |
| 4 | 198.3 | 264–293 | **354.8** (89) | 360 |
| 8 | 255.2 | 348–392 | **464.2** (58) | 469 |

Concurrent prompt processing (pp512, 8 at once): **9,547 tok/s** total.

Single request on the default server (no `GRIMOIRE_SEQ_SLOTS`) is unchanged:

| context | prefill | decode (tg256) |
|---|---:|---:|
| ~0 tokens | ~4,400 tok/s | **195.7 tok/s** |
| 2K tokens | 9,570–9,670 tok/s | **183.4 tok/s** |
| 4K tokens | 9,960–9,990 tok/s | **172.2 tok/s** |

### What changed since v1.4
- **Burst admission** ([a97da18](https://github.com/doopeworld/GRIMOIRE/commit/a97da18), [286ad6e](https://github.com/doopeworld/GRIMOIRE/commit/286ad6e)). When several requests arrive together on an idle server, the scheduler waits up to 8 ms for the rest of the burst. The whole burst is then prefilled in one pass.
  - A lone request waits only if the previous busy period served several requests at once. A single-user chat never waits.
  - This is why c4 and c8 no longer swing between runs.
  - Change the wait with `GRIMOIRE_ADMIT_HOLD_MS` (default 8; 0 = never).
- **Router top-k for every row in one ESIMD launch** ([8744f17](https://github.com/doopeworld/GRIMOIRE/commit/8744f17)). It now matches the single-user decode kernel bit for bit. The old batched kernel spent ~21 µs per layer at every batch size.
- **Small-batch projections read the norm's bf16 rows directly** ([8744f17](https://github.com/doopeworld/GRIMOIRE/commit/8744f17)). The fp32 rows are no longer converted again for every projection.

GPU time per batched decode step on Ornith:

| rows | v1.4 | v1.5 |
|---:|---:|---:|
| 2 | 8.7 ms | 7.9 ms |
| 4 | 11.5 ms | 10.7 ms |
| 8 | 17.0 ms | 16.3 ms |

### Use
```
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.5/grimoire-b70-v1.5.tar.gz
docker load < grimoire-b70-v1.5.tar.gz          # -> grimoire-b70:latest
docker run -d --name grimoire --network host --init --device /dev/dri/renderD128 \
  -e GRIMOIRE_SEQ_SLOTS=8 \
  -v /path/to/models:/models grimoire-b70:latest \
  server --model /models/<checkpoint> --proj mxfp4 --ctx 8192 --port 8000
```
Find your B70's render node with `ls -l /dev/dri/by-path/`. The numbers can change after a reboot. Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

sha256 `1aa2237d2ad2dccaeffa8e1176f322bc83c8332a42554722e0245a4039de140b`
