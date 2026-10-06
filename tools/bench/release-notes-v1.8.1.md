## GRIMOIRE v1.8.1 -- K2 answers stop where they should; two-GPU mode in the image

```
docker load < grimoire-b70-v1.8.1.tar.gz      # -> grimoire-b70:latest
```

### Fixes

**K2-Horizon-MoVA-36B-A4B kept writing after its answer.** K2's tokenizer prefixes every
control token with `ifm|` (`<|ifm|im_start|>`, `<|ifm|im_end|>`, `<|ifm|endoftext|>`). GRIMOIRE
looked for the end-of-turn token only under ChatML's names, found none, and so never stopped a K2
answer. After "Paris" it went on to max_tokens with unrelated text. K2 now:

- stops on `<|ifm|im_end|>` and `<|ifm|endoftext|>` (the two ids in its `generation_config.json`);
- gets its own chat template. It is byte-identical to the checkpoint's Jinja template: BOS first,
  no newline between turns, and the assistant turn opens a `<ifm|think>` block.

llama-benchy on K2 (4 slots, `--ctx 16384`): pp512 919 tok/s, tg128 66.8 at 1 user, 83.4 total at
4 users, coherence test passed.

**A second stop token for every model.** It is whichever exists of `<|eot|>` (Harmony / Muse),
`<|ifm|endoftext|>` (K2) or `<|endoftext|>` (Qwen family, listed next to `<|im_end|>` in their
`generation_config.json`).

### New: two GPUs in one container

`multi` runs one server rank per GPU (pipeline parallel by default, `multi TP` for tensor
parallel). This is for checkpoints that do not fit one B70. Example:
Ornith-1.5-35B-A3B-FP8 on two Arc Pro B70s:

```
docker run -d --init --stop-timeout 300 --ipc=host --shm-size=10g \
    --device /dev/dri/renderD128 --device /dev/dri/renderD131 \
    -v /dev/dri/by-path:/dev/dri/by-path:ro -v /path/to/models:/models -p 8000:8000 \
    -e ZE_AFFINITY_MASK=0,1 -e GRIMOIRE_MULTI_GPUS=2 -e GRIMOIRE_PP_SPLIT=20 -e GRIMOIRE_SEQ_SLOTS=8 \
    -e GRIMOIRE_DEFER_MOE_GATHER=1 -e GRIMOIRE_BF16_QKV=1 -e GRIMOIRE_BF16_DN_QKV=1 \
    grimoire-b70:latest multi --model /models/Ornith-1.5-35B-A3B-FP8 --proj fp8 --ctx 32000 --port 8000
```

Measured: ready in 63 s, 77 tok/s for one 128-token request. `GRIMOIRE_PP_SPLIT` is the number of
layers the first GPU keeps.

### Other models, one 128-token request through the server (v1.8 / v1.8.1)

| Model | `--proj` | Settings | Decode tok/s |
|---|---|---|---:|
| Ornith-1.5-35B-A3B (MXFP4) | `mxfp4` | 8 slots, `--ctx 32000` | 201.8 |
| Qwen3.6-35B-A3B GPTQ-Int4 | `int4` | 8 slots, `--ctx 32000` | 128.7 |
| K2-Horizon-MoVA-36B-A4B | `mxfp4` | 4 slots, `--ctx 16384` | 81-82 |
| Muse-Glimmer-30B INT4-W4A16 | `int4` | one request at a time, `--ctx 16384` | 28.4 |
| Agnes-3.0-Flash | `mxfp4` | 8 slots, `--ctx 16384` | 28.0 |
| Qwen3.8-Flash-Next (tiered) | `bf16` | 112 experts / layer in VRAM, `--ctx 65536` | 22.3 |

Qwen3.8-27B GPTQ + MTP is unchanged from v1.8: llama-benchy tg128 62.9 at 1 user, 186.9 total at
8 users.

### Image

`grimoire-b70-v1.8.1.tar.gz`, 640,834,579 bytes, sha256
`794cee4159a03ab8a693a09ee6d85f6bfd3924a2dfbfdeb62995feeeb4b9d40e`, built from commit d702908 (image id
`0de8c1a47ecb`).
