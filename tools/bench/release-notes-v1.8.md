## GRIMOIRE v1.8 -- Qwen3.8-27B GPTQ-Int4: MTP up to 72 tok/s, prompt 2,070 tok/s

```
docker load < grimoire-b70-v1.8.tar.gz        # -> grimoire-b70:latest
```

Model: [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16),
`--proj int4 --ctx 16384 -e GRIMOIRE_MTP=1 -e GRIMOIRE_SEQ_SLOTS=8`. One Arc Pro B70, llama-benchy
0.4.0, total tokens/s:

| Test | Users | v1.7.1 | **v1.8** |
|---|---|---:|---:|
| pp512 / tg128 | 1 / 2 / 4 / 8 | 49.0 / 67.4 / 84.2 / 139.2 | **53.9 / 87.9 / 121.7 / 187.4** |
| pp4096 / tg128 | 1 / 2 / 4 / 8 | 51.1 / 38.4 / 36.2 / 39.7 | **55.2 / 78.9 / 107.9 / 149.0** |
| prompt pp4096 | 1 | 1,731 | **2,069** |

Every llama-benchy run draws different text from its book, and how much MTP gains depends on the
text: four runs of v1.8 at 1 user gave 53.9 - 71.8 tok/s after pp512 and 55.2 - 63.5 after
pp4096 (the table is the release validation run, `tools/bench/validate_image_v18.sh`).

### What changed

**MTP drafts fixed.** Two differences from how the checkpoint's MTP head is meant to be run:

- Each draft ran at the position of the token it was given instead of the position of the
  hidden state it came from. The head's own cache stores that pair at the hidden state's
  position, so every draft looked at its context shifted by one, with the newest entry stored
  twice.
- The head was fed the hidden state before the model's final norm; it expects the one after it
  (the vector the output layer reads).

**MTP defaults.** `GRIMOIRE_MTP=1` is now enough:

- 4 drafts per step (`GRIMOIRE_MTP_K`, was 3). At 1 user, tg128 after pp512 / pp4096:
  3 drafts 61.9 / 56.0, 4 drafts 63.5 / 69.6, 5 drafts 66.2 / 65.0 tok/s (same session).
- The MTP head is stored in MXFP4. Drafting takes 5.1 instead of 9.2 ms per step. The head
  only proposes tokens: every token you get is still the model's own greedy choice.
  `GRIMOIRE_MTP_HEAD_FMT=bf16` keeps it in BF16.
- Drafting stays on while up to 4 requests are active (`GRIMOIRE_SPEC_MAX_SEQS`, was 2).

**INT4 verify faster.** For 2 to 8 rows (the MTP check and batched decoding), the INT4
matrix kernel now runs with 128 registers per thread and splits the work over more threads:
42.2 -> 38.0 ms per MTP check. The output layer is also checked in one pass instead of row by row.

**INT4 prompt processing.** GPTQ / INT4 weights now go through the fast prompt path that
MXFP4 used, with the feed-forward gate fused in: 1,731 -> 2,069 tok/s at 4,096 tokens, same
output text (Qwen3.8-27B-W4A16, group 64: a 5,987-token prompt 4.72 -> 3.81 s).

**Several long prompts at once.** Two 4,096-token prompts do not fit one prefill call at
`--ctx 16384` with 8 slots, so they are processed one after another. The first user's first token
used to go out before the second prompt was processed and then wait for it, which llama-benchy
counted as slow decoding. First tokens of a burst now go out together: pp4096/tg128 at 2 / 4 users
48.0 / 46.0 -> 78.9 / 107.9 tok/s total.

### Comparison on the same card

vLLM 0.30.1 was run with the flags from the
[B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) for
this checkpoint, in its own container on the same B70. That means MTP with 4 drafts, fp8 KV
cache and GPTQ, with no cookbook patches and the head in BF16. The same llama-benchy runs gave:

| Test | Users | vLLM 0.30.1 | GRIMOIRE v1.8 |
|---|---|---:|---:|
| pp512 / tg128 | 1 / 2 / 4 / 8 | 45.1 / 85.2 / 130.4 / 122.3 | 53.9 - 71.8 / 87.9 / 121.7 / 187.4 |
| pp4096 / tg128 | 1 / 2 / 4 / 8 | 40.1 / 57.8 / 64.3 / 46.0 | 55.2 - 63.5 / 78.9 / 107.9 / 149.0 |
| prompt pp4096 | 1 | 1,948 | 2,069 |

vLLM's MTP accepted 37% of its drafts on llama-benchy's text. The cookbook's own 84-113 tok/s
come from prompts where about 95% of drafts are accepted.

### Still open

- A long prompt that arrives while others are decoding still pauses them for its prefill.
  Mixing prefill chunks into the decoding steps is the next item.
- Tool calling in the server.

### Image

`grimoire-b70-v1.8.tar.gz`, 640,860,913 bytes, sha256
`2d8b5132c246dfb422326f48a079bc95a22c904c602b53c2c514c9435ac5c7a2`, built from commit 05bc4ec
(image id `afe77e3e7ec6`).
