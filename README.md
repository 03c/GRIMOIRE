# GRIMOIRE

![GRIMOIRE LLM inference banner](assets/grimoire-banner.jpg)

**LLM inference built for Intel Arc Pro B70 Battlemage GPUs.**

GRIMOIRE is a native inference engine written in C++ with SYCL and Level Zero. It is being developed to run and optimize large language models directly on Intel Battlemage hardware, with a focus on the Arc Pro B70.

The aim is straightforward: make high-performance local LLM inference possible on this hardware, using kernels and memory paths designed for the GPU instead of relying on a general-purpose inference stack.

> **Project status: Experimental.** GRIMOIRE is actively developed, but it is not yet a stable, production-ready release. Expect incomplete features, changing model support, and performance or compatibility issues. A ready-to-use inference image or downloadable model package will be added here later.

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
| Ornith-1.5-35B-A3B token generation | **198.6 tokens/s** | Single B70; 256 generated tokens in the recorded run. [Commit](https://github.com/doopeworld/GRIMOIRE/commit/9cd4a9191ea4ee704f9214861b3cb1c9802b3291) |
| Ornith-1.5-35B-A3B prefill | **10,030–10,165 tokens/s** | Two-GPU pipeline-parallel run, 5,987-token prompt. Output matched the single-B70 run for the checked text. [Commit](https://github.com/doopeworld/GRIMOIRE/commit/b970875d52cf3e6cea8d354808dd7ab5ff030323) |
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
| Qwen3.8-27B + native MTP head | MXFP4 or GPTQ-Int4 | target | multi-token prediction |
| Qwen3.8-27B-DFlash2 | MXFP4 | speculative draft | for Qwen3.8-27B |
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

Speculative decoding (the MTP and DFlash2 rows above) is currently correct but *slower*
than plain decode on every model it's paired with — functional, not yet a throughput win.

## Getting started

The engine and build scripts are in this repository, along with a `Dockerfile` that
builds a minimal runtime image -- just GRIMOIRE's own binaries plus Intel's GPU driver
stack, nothing else:

```
docker build -t grimoire-b70 .
docker run --init --stop-timeout 300 --device /dev/dri/renderDXXX \
    -v /path/to/your/models:/models -p 8000:8000 \
    grimoire-b70 server --model /models/<checkpoint> --proj mxfp4 --port 8000
```

`--init` and a generous `--stop-timeout` are not optional: GPU work in flight when a
container is killed without them can wedge the card hard enough to need a power cycle.
Replace `server` with `generate -m ... -p "..." -n <tokens>` for a one-shot CLI run
instead of the HTTP server. Check the project status and the model table above before
choosing a model or format.

## Development notes

GRIMOIRE is under active development. Model support, build requirements, and performance may change as the engine evolves. Benchmarks should be read with their linked commit notes, which record the model, test conditions, correctness checks, and known trade-offs.

## Links

- [Source code](https://github.com/doopeworld/GRIMOIRE)
- [Recent commits](https://github.com/doopeworld/GRIMOIRE/commits/main)
- [Issues](https://github.com/doopeworld/GRIMOIRE/issues)
