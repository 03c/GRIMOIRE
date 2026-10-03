# HANDOFF 2026-10-03 — served speed vs CLI speed, concurrency, prefix cache

Live doc: updated as each result lands. Newest section last.
Runs and raw logs: `/mnt/storage/isos/grimoire-runs/bench-1003/` on the Tower.
llama-benchy pinned to **0.4.0** (`uvx llama-benchy@0.4.0`), with `--tokenizer`
pointing at the model directory, so prompt and depth sizes are in the model's
own tokens. The fallback was gpt2 approximations.

## Context

On 2026-10-02, llama-benchy against `GRIMOIRE-ORNITH` (HTTP) measured TG
127.6 tok/s (pp2048/tg256) and 101.3 (pp4096/tg32). The CLI measured
198.6 tok/s on the same model. Ian asked why; the session paused overnight.
A power cut followed. Nothing had been started, and nothing was lost.

## 1. Why served TG was 101–128 when the CLI said 199. SOLVED.

There were three causes, with no HTTP cost to speak of. Every number below
is Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE on gpu0 (03:00.0). The first request to
each server was a short chat, which is what GRIMOIRE-ORNITH saw yesterday.

| config (llama-benchy tg, tok/s)                                 | depth ~0 | 2K    | 4K    |
|-----------------------------------------------------------------|---------:|------:|------:|
| yesterday's container: grimoire-b70, `SYCL_UR_USE_LEVEL_ZERO_V2=0` | 159–163  | 121–124 | 98–99 |
| same image, V2 adapter (env var removed)                         | 194–197  | 141–142 | 109   |
| V2 + `GRIMOIRE_DECODE_SPLITS=128` (workaround for cause b)       | 195–196  | 181–182 | 171–172 |
| CLI, engine only, differential (n160 − n32) at the same depths   | 202.5    | 180.8 | 172.0 |

a. **The legacy Level Zero adapter.** All four GRIMOIRE Unraid templates
   (ORNITH/QWEN/MUSE/DUAL) set `SYCL_UR_USE_LEVEL_ZERO_V2=0`, which forces the
   legacy UR adapter. That setting was inherited from `grimoire:b70-native`,
   which segfaulted on V2 on 09-25. The new `grimoire-b70` image runs V2
   cleanly, and V2 is the oneAPI 2026.1 default and what every CLI baseline
   used. Cost of the legacy adapter: about 1.2 ms per token (194 → 159 tok/s).

b. **The decode graph froze the attention split count at capture.**
   `build_graph()` records `forward(1)` with the host `pos` at its current
   value. `decode_splits()` therefore baked `max(8, ceil(len/32))` for the
   *capture-time* length into the graph. With one sequence slot,
   `bind_seq_slot()` returns early, so `reset()` never invalidates the graph,
   and the server captures it ONCE, on its first request. After a short first
   request, every later request ran its whole context through 8 splits: on
   Ornith that is 64 ESIMD threads on a 256-EU card. Cost: 1.6 ms/token at 2K
   and 3.4 ms/token at 4K. The CLI never showed this, because it captures
   right after the real prompt's prefill.

c. **Apples and oranges.** The CLI's 198.6 is tokens 32..160 after a
   20-token prompt. llama-benchy measures decode after 2K or 4K tokens of
   context. The engine itself is 181 at 2K and 172 at 4K. That is
   attention cost, not serving overhead.

**The HTTP path costs nothing measurable.** Once a and b are removed,
served = CLI within noise at 2K and 4K. At ~0 depth the remaining 2–3% is
the 128-split workaround itself (5.09 vs 4.94 ms/token on the CLI). The real
fix below does not pay that.

**Fix for b (branch `served-decode-fix`):** `AttnParams::capture` is set while
a graph is recorded. The launch is then sized for the whole cache (`seq_cap`),
and every decode-attention kernel (ESIMD hd256, GQA, per-head) plus
`launch_flash_merge` derives, from the *device-side* length, the split count
that direct submission would launch. The surplus splits return at once.
Direct submission is unchanged. A replayed step now does exactly the
direct-submission math at every length. Files: `src/kernels.hpp`,
`src/attention.cpp`, `src/grimoire.cpp` (`ap.capture = recording` in
`forward`, `forward_muse`, `forward_gemma4`).

**Fix for a:** remove `SYCL_UR_USE_LEVEL_ZERO_V2=0` from the templates and the
container (pending: done after validation, see below).

### Validation of the fix (bench-1003/validate_fix.txt, gpu0, 10:05)

- CLI, short context, new binary vs old (`bin/grimoire.pre1003`): n=160 text
  **IDENTICAL**; decode 5.02 ms/token = 199.4 tok/s (old binary today 4.94 ms,
  within run-to-run noise; the gpu1 sweep lane was running alongside).
- CLI, 4K context, n=96: graph replay vs `GRIMOIRE_DECODE_GRAPH=0` direct
  submission **IDENTICAL**. New direct vs old direct **IDENTICAL**, so the
  direct path is untouched.
- Served (host-built `bin/grimoire-server`, V2 adapter, NO split env, short
  first request), llama-benchy 0.4.0:

| tok/s        | depth ~0 (pp128) | 2K (pp2048) | 4K (pp4096) |
|--------------|-----------------:|------------:|------------:|
| tg32         | 196.4            | 183.6       | 171.5       |
| tg256        | 195.1            | 183.4       | 172.4       |
| CLI (engine) | 202.5 / 199.4    | 180.8       | 172.0       |
| 10-02 container | --            | 127.6 (tg256) | 101.3 (tg32) |

Served TG now equals engine TG at every depth. Prefill is unchanged at
9.5–9.8k tok/s (pp2048/pp4096).

## Superseded

- 2026-10-02 memory/notes: "the gap is a fixed per-request/per-token HTTP
  serving cost (SSE framing, JSON encode, harmony split)". **Wrong.** It was
  the adapter plus the frozen split count. Serving costs ~0.
- 2026-10-02 notes: "Ornith/Muse cannot batch: linear attention falls back
  to one request at a time". **Wrong for current code.**
  `batch_unsupported_reason()` only requires `GRIMOIRE_SEQ_SLOTS >= 2`, a
  matrix-capable device and no distributed speculation. The "one request at
  a time" line on GRIMOIRE-ORNITH came from the container being started
  without `GRIMOIRE_SEQ_SLOTS`.
- The host `bin/grimoire-server` was a 09-25 build, older than the ESIMD
  decode work. Anything launched through `tools/serve.sh` before today ran
  that stale server, and with `GRIMOIRE_W4A8=1` and the Ornith-only flags
  hard-coded. Rebuilt today.

## INCIDENT 10:05:30: gpu1 (0b:00.0) dropped off the PCIe bus. Hardware, needs a reboot.

The sweep lane on gpu1 had just started (Qwen3.8-27B-MXFP4, batching server,
first requests after the model load) when the following was logged:

```
10:05:30 pcieport 0000:00:06.2: AER: Multiple Correctable error ... Physical Layer, RxErr
10:05:30 pcieport 0000:0a:02.0: Unable to change power state from D3hot to D0, device inaccessible
10:21:14 xe 0000:0b:00.0: Force wake domain 0: wake. MMIO unreliable (forcewake register returns 0xFFFFFFFF)
```

- `lspci -s 0b:00.0` now returns nothing. The card is gone from config space.
- The neighbouring root port 00:06.0 (the Arc B580's path, 05:00.0) has logged
  correctable RxErr all morning, every 30–60 s. This is the same signature as the
  09-26 drop, when a B70 sat on 00:06.0: the x4 slot area is electrically
  marginal at Gen4 (16 GT/s). The B580 being back on 00:06.0 is the hardware
  change since TP/PP last ran cleanly on gpu0+gpu1 (09-30..10-02).
- gpu0 (03:00.0, root port 00:01.0, Gen4 x8): zero AER errors, still clean.
- The container `z6912` (grimoire-server on the dead card) is stuck spinning in
  the Level Zero driver. Per the GPU hang rule it was **not** killed. Its
  clients were stopped. **Recovery: reboot the Tower.** Before relying on gpu1
  again, force that slot to Gen3 in the BIOS and/or reseat its riser/cable.
- Everything after this runs on gpu0 only.

## 2. Concurrency 1/2/4/8 (sweep running on gpu0: bench-1003/zoo, lane-gpu0.txt)

Method: grimoire-server (host-built `bin/`, V2 adapter), `GRIMOIRE_SEQ_SLOTS=8`, llama-benchy
0.4.0 `--pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2`. A batch check is also run: 4
prompts asked serially, then all 4 at once.

**Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE** (scheduler: "batching up to 8 requests per step"):

| concurrency | total tg tok/s | per request | pp512 TTFT |
|---|---:|---:|---:|
| 1 | 132.7 | 132.7 | 116 ms |
| 2 | 94.7 | 49.5 | 169 ms |
| 4 | 133.4 | 37.1 | 289 ms |
| 8 | 175.3 | 25.8 | 512 ms |
| (one-at-a-time graph path, no SEQ_SLOTS) | **195** | 195 | |

- With `GRIMOIRE_SEQ_SLOTS>=2`, **every** request goes through the batched path
  (`decode_batch` -> `prefill` with M rows). A lone request then decodes at 132 instead
  of 195 tok/s: no graph replay, and none of the M=1 ESIMD decode kernels.
- On this MoE model, 8 concurrent requests deliver less in total than one request on
  the graph path. Each extra row routes to its own experts, so weight traffic grows
  with M instead of being shared (this is a top-8-of-256 MoE).
- Batch check: 2/4 identical. The two that differ are coherent and identical for many
  tokens, then split at a near-tie ("Let me find some." / "Let me find some primes...").
  That is floating-point summation order (M=1 vs M=4 kernels), not cross-talk.

## 3. Prefix cache. It never hits for an HTTP client, on any model (diagnosed)

`prefix_reuse()` (src/grimoire.cpp) reuses a slot only when the slot's **entire**
cached token list (prompt + reply, minus the last token) is a prefix of the new
prompt. There is no partial match. For a hybrid (DeltaNet) model, partial reuse is
impossible as built, because the recurrent state is kept only at the END of the
cached sequence. Three client patterns, measured on Ornith (cache off -> on, 2nd request):

| pattern | off | on | why it cannot hit |
|---|---:|---:|---|
| shared ~2K system context, new user msg (llama-benchy `--enable-prefix-caching`) | 0.51 s | 0.51 s | the cached sequence contains the OLD user msg + reply |
| multi-turn chat, turn 2 = turn 1 + reply + new question | 0.51 s | 0.53 s | the generation prompt is `<|im_start|>assistant\n<think>\n` (tokenizer.cpp:638), but history renders the assistant turn WITHOUT `<think>\n` (:635), so it diverges right after `assistant\n` |
| raw /v1/completions continuation (prompt + completion + more) | 0.45 s | 0.46 s | the reply re-tokenizes differently: the model emitted 3 separate `\n` tokens, re-encoding the text gives one `ĊĊĊ` token (24 generated ids vs 22 on re-encode), so the full-sequence match fails |

Answers were identical with the cache on and off (no corruption). llama-benchy's
prefix test shows no change at all (`pp256 @ d4096` TTFT 492 ms off, 495 ms on).

**What would make it work (design, not done tonight):**
1. Keep a recurrent-state checkpoint at the END OF THE LAST USER MESSAGE: prefill the
   prompt minus the generation header, snapshot (DeltaNet state + conv ring, ~63 MB on
   Ornith; KV rows are already in place), then prefill the header. A turn-2 prompt
   always contains turn 1 up to that point, token-exact. Only the old reply plus the
   new message gets prefilled. That makes multi-turn chat and agents hit.
2. Match the longest common prefix that lands on a checkpoint, not the whole cached
   list. Attention-only models can truncate KV at ANY position, so add partial reuse
   for them directly.
3. Optionally checkpoint after the system message too, so llama-benchy's
   shared-context pattern (and agents with one long system prompt) hits.

## Release v1.1, published 2026-10-03 11:42 (CEST)

https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.1 (pre-release, target c08ec32):
`grimoire-b70-v1.1.tar.gz`, 640,285,691 bytes, sha256 `8d1030bc028957a38142ba3e00bdfd152105a0a3c5d0c743e950917bd067fb97`.
Validated before upload with llama-benchy on gpu0, run like GRIMOIRE-ORNITH without `V2=0`.
Coherence PASSED. tg256: 195.6 / 183.5 / 172.5 tok/s at depth ~0 / 2K / 4K; pp 9.6–9.9k.
v1 is left as it was. Tool: `tools/bench/upload_release.py` (token from env `GH_TOKEN`).
The GRIMOIRE-ORNITH Unraid template was fixed: `--device /dev/dri/renderD128` (gpu0 after
the 10-03 reboot; renderD129 is now the iGPU), and the `SYCL_UR_USE_LEVEL_ZERO_V2` line was
removed. The backup is in bench-1003/templates-backup/. QWEN/MUSE/DUAL still run the old
`grimoire:b70-native` image, which needs `V2=0`, so they were left alone. Move them to
`grimoire-b70:latest` before removing it.


## 4. Full sweep, 23 checkpoints (bench-1003/zoo, zoo-table.md). gpu0, 10:58–14:30

Server: `GRIMOIRE_SEQ_SLOTS=8`, prefix cache off (phase 1) / on (phase 2). c1..c8 = total decode tok/s
(llama-benchy pp512/tg64). "batch==serial" = 4 prompts serially vs all at once (greedy); the
non-identical ones are coherent and split at near-ties. The prefix columns are the 2nd request's
wall time, cache off -> on. Muse GPTQ/MXFP4 ran c1/c2 only (each Muse run took 30–43 min).

| model | fmt | sched | c1 | c2 | c4 | c8 | c8/c1 | batch==serial | pfx chat off→on (s) | pfx raw off→on (s) | benchy pp256@d4096 TTFT off→on (ms) |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| Agnes-3.0-Flash | mxfp4 | batch | 25.7 | 19.1 | 32.0 | 48.6 | 1.89x | 4/4 | 2.61→2.63 | 2.30→2.29 | 2570→2571 |
| K2-Horizon-MoVA-36B-A4B | mxfp4 | batch | 63.6 | 42.8 | 48.2 | 52.8 | 0.83x | 0/4 | 1.33→1.33 | 1.17→0.49 | 1227→1227 |
| Muse-Glimmer-30B-GPTQ-INT4 | int4 | batch | 0.8 | 1.6 | - | - | - | 4/4 | 43.56→43.55 | 32.73→32.72 | 7050→7035 |
| Muse-Glimmer-30B-INT4-W4A16 | int4 | batch | 0.8 | 1.6 | 3.0 | 5.6 | 6.89x | 4/4 | 43.57→43.55 | 32.72→32.72 | 7055→7046 |
| Muse-Glimmer-30B-MXFP4 | mxfp4 | batch | 0.8 | 1.5 | - | - | - | 4/4 | 44.44→44.45 | 33.35→33.34 | 6947→6949 |
| Ornith-1.5-35B-A3B-FP8 | mxfp4 | batch | 132.5 | 95.4 | 134.2 | 175.9 | 1.33x | 4/4 | 0.51→0.53 | 0.45→0.46 | 490→492 |
| Ornith-1.5-35B-A3B-GPTQ-Int4 | int4 | batch | 83.7 | 40.4 | 33.4 | 30.6 | 0.37x | 4/4 | 2.21→2.22 | 2.10→2.11 | 1963→1964 |
| Ornith-1.5-35B-A3B-INT4-W4A16-AutoRound | int4 | batch | 83.9 | 41.4 | 33.8 | 31.2 | 0.37x | 4/4 | 2.20→2.21 | 2.08→2.09 | 1956→1947 |
| Ornith-1.5-35B-A3B-MTPFIX | mxfp4 | batch | 132.6 | 94.6 | 135.5 | 175.2 | 1.32x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→497 |
| Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE | mxfp4 | batch | 132.7 | 94.7 | 133.4 | 175.3 | 1.32x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→495 |
| Ornith-1.5-35B-A3B-NVFP4 | mxfp4 | batch | 131.3 | 95.8 | 131.8 | 176.6 | 1.35x | 1/4 | 0.51→0.53 | 0.46→0.46 | 496→491 |
| Ornith-1.5-35B-A3B | mxfp4 | batch | 132.8 | 96.2 | 133.6 | 177.3 | 1.34x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→492 |
| Qwen3.6-35B-A3B-GPTQ-Int4 | int4 | batch | 83.7 | 42.2 | 34.5 | 32.0 | 0.38x | 4/4 | 2.17→2.18 | 2.05→2.06 | 2014→2011 |
| Qwen3.8-27B-FP8 | mxfp4 | batch | 30.6 | 22.2 | 37.4 | 57.5 | 1.88x | 4/4 | 2.17→2.18 | 1.90→1.90 | 2131→2129 |
| Qwen3.8-27B-GPTQ-Int4-MTP-BF16 | int4 | batch | 19.1 | 16.8 | 17.0 | 16.5 | 0.87x | 4/4 | 3.10→3.12 | 2.67→2.68 | 2569→2580 |
| Qwen3.8-27B-MXFP4-AutoRound | mxfp4 | batch | 30.6 | 22.3 | 37.6 | 57.7 | 1.88x | 3/4 | 2.17→2.18 | 1.90→1.90 | 2137→2131 |
| Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 | mxfp4 | batch | 30.6 | 22.2 | 37.7 | 57.6 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.91 | 2129→2132 |
| Qwen3.8-27B-MXFP4-GRIMOIRE | mxfp4 | batch | 30.6 | 22.3 | 37.8 | 57.5 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.91 | 2133→2131 |
| Qwen3.8-27B-NVFP4 | mxfp4 | batch | 30.6 | 22.2 | 37.4 | 57.5 | 1.88x | 4/4 | 2.16→2.18 | 1.90→1.90 | 2129→2129 |
| Qwen3.8-27B-W4A16 | int4 | batch | 19.0 | 16.7 | 16.9 | 16.5 | 0.87x | 4/4 | 3.12→3.13 | 2.69→2.69 | 2598→2587 |
| Qwen3.8-27B-int4-AutoRound | int4 | batch | 19.0 | 16.8 | 16.9 | 16.5 | 0.87x | 4/4 | 3.10→3.12 | 2.68→2.68 | 2576→2579 |
| Qwen3.8-27B | mxfp4 | batch | 30.6 | 22.3 | 37.7 | 57.5 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.90 | 2128→2131 |
| Qwen3.8-Flash-Next-NVFP4 | bf16 | batch | - | - | - | - | - | 0/4 | ERR | ERR | - |

- Every model batches without errors, except Flash-Next: every request failed with HTTP 500
  "batched decode step failed", because its file-backed PLE table supports one sequence only.
  **Fixed (served-tools):** `batch_unsupported_reason()` now names it, and the scheduler
  falls back to one at a time.
- Batching scales poorly everywhere (c8/c1 = 0.37x–1.89x; an M=4 step costs ~5.5x an M=1 graph
  step on Qwen3.8-27B). int4 checkpoints are worst: aggregate FALLS with concurrency. Cause: the
  batched decode IS prefill() (prompt-sized GEMMs, per-row conv/DeltaNet/attention, no graph,
  host syncs), and admission prefills stall every live row.
- Prefix cache: no hit anywhere except K2's raw continuation (1.17 -> 0.49 s). See section 3.

## 5. Scheduler solo fast path. Validated, branch served-tools

`decode_solo()`/`solo_ok()`: when exactly ONE sequence is live (and no PP/TP/DAG/Qwen4-Exp/
drafter), the scheduler binds its slot, puts cursor + token on the device and replays the
decode graph. decode_batch() re-records the graph after any multi-row step.
`GRIMOIRE_SCHED_SOLO=0` turns it off.

| with GRIMOIRE_SEQ_SLOTS=8 | lone request output vs one-at-a-time | c1 before -> after | c2..c8 |
|---|---|---|---|
| Ornith-1.5-35B-A3B-MXFP4 | IDENTICAL (200 + 60 tokens); overlap test IDENTICAL | 132.7 -> **193.3** | unchanged (95 / 134 / 176) |
| Qwen3.8-27B-MXFP4 | IDENTICAL | 30.6 -> **33.1** | unchanged |
| Muse-Glimmer-30B-INT4-W4A16 | IDENTICAL | 0.8 -> **18.9** | unchanged (batched Muse path still ~0.8/row) |

Ornith at depth, solo build, SEQ_SLOTS=8: tg256 196.3 / 184.5 / 173.2 at depth ~0 / 2K / 4K, the
same as the one-at-a-time server. Turning concurrency on no longer costs single-user speed.

## Release v1.2, published 2026-10-03 14:55 (CEST). State at end of day

https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.2 (pre-release, target b745052):
`grimoire-b70-v1.2.tar.gz`, 640,343,202 bytes, sha256 `8a3a2c760d77672a57f89a8d27fa2b3e040b45882b13ba84f83630d03e84a7db`.
Validated before upload (bench-1003/validate_image_v12.txt), Ornith on gpu0:
- Default: coherence PASSED, tg256 195.6 / 183.5 / 172.4 at depth ~0 / 2K / 4K (same as v1.1).
- `GRIMOIRE_SEQ_SLOTS=8`: tg64 c1 193.4 (v1.1: 132.7), c2 95.2, c4 132.6, c8 174.2.

Docker images on the Tower: `grimoire-b70:latest` = v1.2 (48df3e7c6b26), `:v1.1` (322cb487f99c), `:v1` (9270d65a31c3).

**GRIMOIRE-ORNITH** was recreated on v1.2: port 6889, `--device /dev/dri/renderD128` (gpu0),
no `SYCL_UR_USE_LEVEL_ZERO_V2`, one request at a time (best single-user speed). Healthy.

**Open, for Ian:**
1. Reboot the Tower to bring gpu1 (0b:00.0) back. Container `z6912` is stuck on the dead card
   and goes away with the reboot. Before loading gpu1 again, force its slot (root port 00:06.2)
   to Gen3 in the BIOS or reseat it.
2. After any reboot, check `ls -l /dev/dri/by-path/`. Render nodes renumber, and the Unraid
   templates pin `renderD128`.
3. QWEN/MUSE/DUAL templates still use the old `grimoire:b70-native` image (it needs `V2=0`).
   Move them to `grimoire-b70:latest` and drop `V2=0` together.

## 6. Profile of one batched decode step (Ornith, gpu0, 2026-10-03 ~15:10 CEST). PAUSED here

`GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_SCHED_SOLO=0 GRIMOIRE_TIME_LAYER=all`, M short concurrent
requests. Each region is synced, so absolute times are inflated; compare the ratios.
Driver: bench-1003/prof_batch.py.

| region (ms per step) | M=1 | M=2 | M=4 | M=8 |
|---|---:|---:|---:|---:|
| routed MoE | 2.16 | 7.21 | 11.11 | **16.46** |
| post norm + route | 1.50 | 1.70 | 2.02 | 2.63 |
| DN qkv / z / gate / out projections | 2.21 | 5.28 | 5.49 | 5.86 |
| shared expert FFN + shared expert | 1.08 | 2.34 | 2.60 | 3.10 |
| attn q / kv projections | 0.42 | 1.28 | 1.29 | 1.29 |
| final norm + logits | 0.49 | 1.23 | 1.25 | 1.28 |
| DN recurrence + conv + attn flash + rope/kv (per-row loops) | 1.28 | 1.99 | 3.13 | 5.57 |
| TOTAL (timed) | 10.21 | 22.42 | 28.51 | 37.76 |

Uninstrumented wall time (solo on): M=2 23 ms, M=4 32 ms, M=8 48.5 ms per step, including
the short prefills. The solo graph step is ~5.2 ms.

Reading:
- **Routed MoE is 44% of the M=8 step** and grows ~7x from M=1, the same factor as the
  distinct-expert bytes (8 tokens x top-8 hits ~57 distinct experts/layer). It runs at about
  the same ~40% of bandwidth as at M=1. Lever: a grouped MoE kernel at ~80% of bandwidth
  (16.5 -> ~8 ms).
- **The projections jump 2.5x from M=1 to M=2 and then stay flat** (weights read once, but
  a slower small-M GEMM path than the M=1 ESIMD GEMV). Lever: an M-row ESIMD GEMV (~14 -> ~6 ms).
- **The per-row loops grow linearly** (1.3 -> 5.6 ms). Lever: one launch over all rows.
- Plus graph replay per batch size (launch overhead). Estimated M=8 step ~14–16 ms ->
  **~500–570 tok/s at c8**. 800 needs ~10 ms per step, i.e. MoE near the bandwidth roofline.

**State at pause:** GRIMOIRE-ORNITH is STOPPED (stopped for this profile). Start it again
from the Unraid template (fixed: renderD128, v1.2 image), or ask Claude to. No GPU work running.
