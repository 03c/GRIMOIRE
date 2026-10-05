## GRIMOIRE v1.7 -- INT4 (GPTQ / AutoRound / W4A16) and speculative decoding on Qwen3.8-27B

Load the image:

```
docker load < grimoire-b70-v1.7.tar.gz      # -> grimoire-b70:latest
```

The README has a per-model launch table (arguments, how to turn on MTP / DFlash where it is
faster, measured speed).

### What changed

**INT4 checkpoints are fast now (GPTQ-Int4, AutoRound, W4A16).** Until v1.6 every INT4 weight
ran on a generic SIMT GEMV. v1.7 has GRIMOIRE's own ESIMD INT4 kernels:

- single-user decode: an ESIMD GEMV that keeps the exact (q - z) * scale dequant;
- 2-16 rows (several users, and the speculative verify pass): a DPAS GEMM on the matrix
  engine with INT4 weights and bf16 activations.

Measured with llama-benchy 0.4.0 (512-token prompts, 128 generated tokens, total tokens/s,
`GRIMOIRE_SEQ_SLOTS=8`), one Arc Pro B70:

| Checkpoint | v1.6: 1 / 2 / 4 / 8 users | v1.7: 1 / 2 / 4 / 8 users |
|---|---|---|
| Qwen3.8-27B GPTQ-Int4, group 128 ([bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)) | ~20 at 1 user (same INT4 kernels as the row below) | **34.8 / 48.1 / 86.4 / 145.3** |
| Qwen3.8-27B W4A16, group 64 (an AutoRound export of Qwen/Qwen3.8-27B) | 19.9 / 18.3 / 19.5 / 19.5 | **33.2 / 45.5 / 82.4 / 139.5** |

Single-user text is identical to the old kernels (checked over 160 greedy tokens on both
checkpoints).

**MTP and DFlash are faster than plain decode on Qwen3.8-27B.** The verify pass now runs on
the small-M DPAS GEMMs by default (it used a slower path before; `GRIMOIRE_SMALLM_DPAS=0`
restores it).

Qwen3.8-27B, one user, llama-benchy tg128 (plain decode for comparison):

| Checkpoint | Plain | MTP (`-e GRIMOIRE_MTP=1 -e GRIMOIRE_MTP_K=3`) | DFlash2 ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), `--dflash-model`) |
|---|---|---|---|
| GPTQ-Int4 + BF16 MTP head ([SergiioB/...-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16), Swift-1.5) | 34.8 | **51.3 - 54.5** | |
| FP8 ([Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), `--proj mxfp4`) | 33.1 | **48.5** (v1.6: 17.6) | **57.1** |

On v1.6 the same recipes ran at 30.1 (MTP) and 28.4 (DFlash2), below plain decode.

**FP8 checkpoints: the MTP head loads with its block scales.** The MTP loader looked its
tensors up without the FP8 `weight_scale_inv` tiles, so an FP8 checkpoint's head drafted
garbage (MTP ran at half the speed of plain decode). Fixed for the dense head and for MoE
heads' experts.

**Muse-Glimmer-30B INT4:** single-user decode 18.9 -> 26.9 tok/s (the INT4 kernel), and
with `GRIMOIRE_SEQ_SLOTS=8` two and four users no longer collapse (1.6 / 3.0 -> 21.7 / 26.6
total tok/s) -- batching still does not add throughput on Muse, so run it single-user.

### Notes

- `GRIMOIRE_MTP_DRAFT_VOCAB` (an old recipe setting, not used by this image) only limits the
  tokens the MTP *draft* can propose. Every emitted token is the target model's choice over
  the full vocabulary, so it can never cut tool-call or other high-id tokens from the output.
  The recipes in the README do not set it.
- The server does not support tool calling yet: requests with `tools` / `tool_choice` are
  rejected with HTTP 400.
- Known issue, fixed in the next release: the server's text decoder hides every added token,
  including Qwen's `<think>`, `</think>`, `<tool_call>` and `</tool_call>`, so clients that
  parse tool calls or the reasoning boundary out of the text cannot find them.

### Image

`grimoire-b70-v1.7.tar.gz`, 640,750,822 bytes, sha256
`a89ce0b7a7128f79554ada74102857fbf198355624c7ee4c19a70edbb8fb63bf`, image id `0ea7472b81e4`,
built from commit 191e05b.
