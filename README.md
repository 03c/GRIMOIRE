9c403c8 Prefill VRAM budget; MTP with many users: adaptive speculation, batched admission
# GRIMOIRE

![GRIMOIRE LLM inference banner](assets/grimoire-banner.jpg)

**LLM inference built for Intel Arc Pro B70 Battlemage GPUs.**

GRIMOIRE is a native inference engine written in C++ with SYCL and Level Zero. It is being developed to run and optimize large language models directly on Intel Battlemage hardware, with a focus on the Arc Pro B70.

The aim is straightforward: make high-performance local LLM inference possible on this hardware, using kernels and memory paths designed for the GPU instead of relying on a general-purpose inference stack.

> **Project status: Experimental.** GRIMOIRE is actively developed, but it is not yet a stable, production-ready release. Expect incomplete features, changing model support, and performance or compatibility issues. Ready-to-use Docker images are on the [Releases](https://github.com/doopeworld/GRIMOIRE/releases) page; [Run a model](#run-a-model) below lists every model with its Hugging Face download, launch flags and measured speed.

## Run a model

### 1. Get the image

```
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.7.1/grimoire-b70-v1.7.1.tar.gz
docker load < grimoire-b70-v1.7.1.tar.gz        # -> grimoire-b70:latest
```

All releases are on the [Releases](https://github.com/doopeworld/GRIMOIRE/releases) page. You can also
build the image yourself: `docker build -t grimoire-b70 .`

### 2. Download a model

Every number in this README was measured with a checkpoint from the table below. Download the
Hugging Face repository into a folder, for example:

```
hf download Qwen/Qwen3.8-27B-FP8 --local-dir /path/to/models/Qwen3.8-27B-FP8
```

### 3. Find the B70's render node

```
ls -l /dev/dri/by-path/      # the B70 is the pci-<its PCI address>-render entry, e.g. renderD128
```

The number can change after a reboot, so check again after every reboot.

### 4. Launch

```
docker run -d --name grimoire --init --stop-timeout 300 \
    --device /dev/dri/renderD128 \
    -v /path/to/models:/models -p 8000:8000 \
    -e GRIMOIRE_SEQ_SLOTS=8 \
    grimoire-b70:latest \
    server --model /models/Qwen3.8-27B-FP8 --proj mxfp4 --ctx 16384 --port 8000
```

Only three things change from one model to another: the `--model` folder, `--proj`, and the extra
flags in the table. The server is OpenAI-compatible at `http://<host>:8000/v1`.

- `--init` and `--stop-timeout 300` are required: killing a container with GPU work in flight can
  drop the card off the PCI bus until a power cycle.
- `-e GRIMOIRE_SEQ_SLOTS=8` lets up to 8 requests decode together. Without it the server answers
  one request at a time.
- `--ctx` is reserved for every slot, so keep it as small as you need. On Qwen3.8-27B with 8
  slots, `--ctx 32000` takes 8.4 GB of KV cache and leaves room for only ~1,500 prompt tokens per
  prefill pass (longer prompts are split, which is slower); `--ctx 16384` leaves ~5,900. The
  server prints the figure at startup: `prefill chunk up to N tokens per call`.
- Decoding is greedy (leave `temperature` out or send 0). Tool calling (`tools`), `stop`,
  `logprobs` and `response_format` are not supported yet; requests that use them get HTTP 400.

### Models, downloads, launch flags and measured speed

Measured on one Arc Pro B70 with llama-benchy 0.4.0, release v1.7 / v1.7.1 unless marked (v1.6 = not
re-measured; the code paths for that row did not change). Decode = generated tokens/s; "8 users" is
the total over 8 concurrent requests with `-e GRIMOIRE_SEQ_SLOTS=8`; prompt = prefill tokens/s at one
user.

| Model | Download from Hugging Face | `--proj` | Extra flags | Decode, 1 user | Decode, 8 users | Prompt |
|---|---|---|---|---:|---:|---:|
| **Qwen3.8-27B** + MTP | [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16) | `int4` | `-e GRIMOIRE_MTP=1 -e GRIMOIRE_MTP_K=3`, `--ctx 16384` | **49 - 54** | **139** <sup>a</sup> | 1,731 (pp4096) |
| Qwen3.8-27B, plain | same | `int4` | | 34.8 <sup>b</sup> | 145.3 <sup>b</sup> | 870 (pp512) |
| **Qwen3.8-27B** + DFlash2 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and the draft [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `mxfp4` | `--dflash-model /models/Qwen3.8-27B-DFlash2` (server argument; measured without `GRIMOIRE_SEQ_SLOTS`) | **57.1** | -- | 1,294 (pp512) |
| Qwen3.8-27B + MTP | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `mxfp4` | `-e GRIMOIRE_MTP=1 -e GRIMOIRE_MTP_K=3` | 48.5 | -- | 1,415 (pp512) |
| Qwen3.8-27B, plain | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) or the BF16 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `mxfp4` | | 33.1 | **162.2** | 1,412 (pp512), ~2,050 (pp4096) |
| **Ornith-1.5-35B-A3B** | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (BF16), [-FP8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8) or [-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) | `mxfp4` | | **193** | **460 - 466** | 7,000 - 7,400 (pp512) |
| Ornith-1.5-35B-A3B, GPTQ | [SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2](https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2) | `int4` | | 100 (v1.6) | 122 (v1.6) | 308 (v1.6) |
| Qwen3.6-35B-A3B, GPTQ | [palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4](https://huggingface.co/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4) | `int4` | | 100 (v1.6) | 122 (v1.6) | 322 (v1.6) |
| Muse-Glimmer-30B | [dudeman2512/Muse-Glimmer-30B-INT4-W4A16](https://huggingface.co/dudeman2512/Muse-Glimmer-30B-INT4-W4A16) | `int4` | leave out `GRIMOIRE_SEQ_SLOTS` | 26.9 | -- | 702 (pp512) |
| Muse-Glimmer-30B | [olka-fi/Muse-Glimmer-30B-MXFP4](https://huggingface.co/olka-fi/Muse-Glimmer-30B-MXFP4) | `mxfp4` | leave out `GRIMOIRE_SEQ_SLOTS` | 24.8 (v1.6) | -- | |
| Agnes-3.0-Flash | [Agnes-AI/Agnes-3.0-Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) | `mxfp4` | | runs; not benchmarked yet | | |
| K2-Horizon-MoVA-36B-A4B | [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) | `mxfp4` | | runs; not benchmarked yet | | |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | `bf16` | `-e GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 -e GRIMOIRE_PLE_FILE=/models/flash-next-ple.bin`, see [FLASH-NEXT-TIERED.md](FLASH-NEXT-TIERED.md) | runs (too large for one B70: VRAM + RAM + SSD); not benchmarked yet | | |

<sup>a</sup> Release v1.7.1, MTP with `GRIMOIRE_SEQ_SLOTS=8`, pp512/tg128: 49.0 / 67.4 / 84.2 / 139.2 tok/s at
1 / 2 / 4 / 8 users -- MTP drafts while at most 2 requests are active (`GRIMOIRE_SPEC_MAX_SEQS`)
and switches to plain batched decoding above that. With 4,096-token prompts: 51.1 / 38.4 / 36.2 /
39.7 (long prompts from several users are still prefilled one after another on this model).
<sup>b</sup> Measured on [bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16),
a Qwen3.8-27B fine-tune in exactly the same format (same kernels, same speed).

**Fine-tunes in the same format work the same way** (`--proj int4`, MTP flags as above). Tested:
[bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
(MTP 51.3 tok/s) and
[greglechin/Swift-1.5-Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/greglechin/Swift-1.5-Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16)
(MTP 47.2 tok/s, CLI).

**Speculative decoding (MTP / DFlash2).**

- **MTP** uses the checkpoint's own multi-token-prediction head, so there is nothing extra to
  download: add `-e GRIMOIRE_MTP=1 -e GRIMOIRE_MTP_K=3`. The head drafts 3 tokens per step (K=3
  measured best: K=2 46.3, K=3 48.4, K=4 39.0 tok/s).
- **DFlash2** is a separate draft model: download it next to the checkpoint and add
  `--dflash-model /models/<draft folder>` after `server`.
- Today both are faster than plain decoding on Qwen3.8-27B only. On Ornith they are not faster
  yet, so leave them off there.

**Prefill (prompt) speed on INT4.** GPTQ / INT4 checkpoints prefill more slowly than MXFP4 for
now (1,746 vs ~2,050 tok/s at 4,096 tokens on Qwen3.8-27B): the fast prefill kernel exists for
MXFP4 only. An INT4 prefill kernel is the next item. For the fastest prompt processing, use the
FP8 or BF16 release with `--proj mxfp4`.

### About the `-MXFP4-GRIMOIRE` folders in older results

Some older results and commit notes name `Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE` or
`Qwen3.8-27B-MXFP4-GRIMOIRE`. These are **not on Hugging Face**: they are pre-converted copies
made on the developer's machine with GRIMOIRE's offline converter (`tools/b70_compile_model.cpp`)
from the official BF16 releases ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). You do not need them: loading the
BF16 (or FP8 / NVFP4) release with `--proj mxfp4` quantizes to the same MXFP4 format at startup
and runs at the same speed (measured: 193 tok/s on Ornith from the FP8 and NVFP4 releases and
from the converted copy). The converted copy only starts faster.

## Why GRIMOIRE was created

Most inference tooling and performance guidance is centered on other GPU platforms. Battlemage owners need an engine that treats Intel hardware as a first-class target and makes its real performance measurable.

GRIMOIRE began as a ground-up inference project for the Arc Pro B70. Building the engine directly exposed important details that a port or a high-level wrapper could hide: how the GPU represents quantized weights, where attention kernels can go wrong, how to keep decode bound by useful memory traffic, and what multi-GPU communication actually costs on this setup.

The project exists to turn those findings into working inference code, reproducible checks, and practical performance improvements for Battlemage users.

## What it is for

- Running supported large language models locally on Intel Arc Pro B70 GPUs.
- Exploring native C++ / SYCL / Level Zero kernels for model inference.
- Measuring prefill and token-generation performance on real hardware.
- Improving single-GPU execution and developing multi-GPU paths for models that need more memory.
- Checking changes against a growing set of model and checkpoint combinations.

GRIMOIRE is a hardware-focused inference engine, not a hosted model service. Model compatibility and performance depend on the architecture, weight format, configuration, and hardware available.

## How it is built

- **C++** inference engine
- **SYCL** GPU programming
- **Level Zero** device runtime
- **Intel Battlemage**, with the Arc Pro B70 as the primary target
- Custom GPU work for operations such as attention, matrix multiplication, quantized weight handling, and mixture-of-experts layers

The project aims to keep inference native to the target hardware and does not use vLLM, PyTorch, or OpenVINO as its inference backend.

## Recent measured results

These are results recorded in the linked project commits on the developer's Arc Pro B70 system. They are examples from specific models and test runs, not universal performance guarantees.

| Workload | Recorded result | Context |
|---|---:|---|
| Ornith-1.5-35B-A3B token generation | **198.6 tokens/s** | Single B70, command-line run, 256 generated tokens; model: the MXFP4 conversion of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (see [Run a model](#run-a-model)). Through the HTTP server with llama-benchy: 193 tokens/s. [Commit](https://github.com/doopeworld/GRIMOIRE/commit/9cd4a9191ea4ee704f9214861b3cb1c9802b3291) |
| Ornith-1.5-35B-A3B prefill | **10,030–10,165 tokens/s** | Two-GPU pipeline-parallel run, 5,987-token prompt, same model. Output matched the single-B70 run for the checked text. [Commit](https://github.com/doopeworld/GRIMOIRE/commit/b970875d52cf3e6cea8d354808dd7ab5ff030323) |
| Qwen3.8-27B with MTP / DFlash2 | **49–54 / 57 tokens/s** | Single B70, one user, llama-benchy through the HTTP server, releases v1.7 / v1.7.1 (plain decoding: 33–35); 139 tokens/s total at 8 users with MTP on. Exact checkpoints and flags in [Run a model](#run-a-model). [Release v1.7.1](https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.7.1) |
| TP decode communication batching | **59.3–60.0 tokens/s** | Recorded TP run after combining independent projection gathers; see the commit for the setup and comparison. [Commit](https://github.com/doopeworld/GRIMOIRE/commit/33757bb631606293a3af87ab76409c479a3c3015) |

The two-GPU figures demonstrate an actively developed path. Tensor and pipeline parallelism are intended for models that need multiple GPUs for capacity; they are not automatically faster for a model that already fits on one card.

Results change as kernels and test coverage evolve. The commit links include the measurements, validation notes, and limitations for each change. The latest model-zoo audit reports **27 confirmed working model/checkpoint combinations** in the local regression sweep, with additional formats and configurations correctly identified as out of scope. [Latest audit](https://github.com/doopeworld/GRIMOIRE/commit/994072e2b75362071eb40aa39455b6a0476c9f2d)

## Supported models

Everything below is validated end-to-end — loaded, generating coherent text, checked
against a reference where one exists — on one Intel Arc Pro B70, by the project's own
regression suite. That suite runs after every change that touches shared decode, prefill,
attention, or MoE code, not as a one-time check.

| Model | Formats | Role | Notes |
|---|---|---|---|
| Ornith-1.5-35B-A3B | MXFP4, NVFP4, FP8, GPTQ-Int4, INT4 (AutoRound W4A16), bf16 source | target | 35B MoE, 256 experts / top-8. 198.6 tok/s decode, 10,030–10,165 tok/s prefill (2×B70), both above |
| Ornith-1.5-35B-A3B-DFlash2 | MXFP4 | speculative draft | for Ornith-1.5-35B-A3B |
| Qwen3.8-27B | MXFP4 (two independent conversions), NVFP4, FP8, W4A16, GPTQ-Int4, INT4 (AutoRound), bf16 source | target | reference point for the vLLM/OpenVINO int4-ov baseline this project compares against |
| Qwen3.8-27B + native MTP head | GPTQ-Int4 (BF16 head), FP8 | target | multi-token prediction, faster than plain decoding since v1.7 |
| Qwen3.8-27B fine-tunes: Swift-1.5, Swift-1.5 Uncensored | GPTQ-Int4 + BF16 MTP head | target | same format as Qwen3.8-27B GPTQ-Int4 |
| Qwen3.8-27B-DFlash2 | MXFP4 | speculative draft | for Qwen3.8-27B ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)); faster than plain decoding since v1.7 |
| Qwen3.6-35B-A3B-GPTQ-Int4 | GPTQ-Int4 | target | 35B MoE |
| Qwen3.6-35B-A3B-DFlash | bf16 | speculative draft | for its own GPTQ-Int4 target |
| Qwen3.5-35B-A3B-DFlash | bf16 | speculative draft | for Ornith, which shares its base architecture |
| Qwen3.8-Flash-Next | NVFP4 | target | too large for one B70's VRAM; tiered across VRAM, system RAM, and SSD |
| Muse-Glimmer-30B | INT4 (W4A16), GPTQ-INT4, MXFP4 | target | hybrid sliding-window / full attention |
| K2-Horizon-MoVA-36B-A4B | MXFP4 (quantized on load from its bf16 release) | target | 36B MoE, its own architecture rather than a Qwen3.5-MoE derivative |
| Agnes-3.0-Flash | MXFP4 | target | |

All seven weight formats (BF16, FP8 E4M3/E5M2, INT8, INT4, MXFP8, MXFP4) run through one
decode path shared by every model above and by the host-side tests.

Tensor-parallel and pipeline-parallel both run correctly across two B70s, but they exist
for checkpoints whose own footprint doesn't fit one card's VRAM — every model in the
table fits a single B70 and runs fastest that way.

Speculative decoding is faster than plain decoding on Qwen3.8-27B since v1.7 (MTP ~1.5x,
DFlash2 ~1.7x for one user). On the other models it is not faster yet, so leave it off there.

## Development notes

GRIMOIRE is under active development. Model support, build requirements, and performance may change as the engine evolves. Benchmarks should be read with their linked commit notes, which record the model, test conditions, correctness checks, and known trade-offs.

## Links

- [Source code](https://github.com/doopeworld/GRIMOIRE)
- [Recent commits](https://github.com/doopeworld/GRIMOIRE/commits/main)
- [Issues](https://github.com/doopeworld/GRIMOIRE/issues)
