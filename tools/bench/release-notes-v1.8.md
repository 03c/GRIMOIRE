## GRIMOIRE v1.8 -- Qwen3.8-27B GPTQ-Int4: MTP 63-70 tok/s, prompt 2,078 tok/s

```
docker load < grimoire-b70-v1.8.tar.gz        # -> grimoire-b70:latest
```

Model: [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16),
`--proj int4 --ctx 16384 -e GRIMOIRE_MTP=1 -e GRIMOIRE_SEQ_SLOTS=8`. One Arc Pro B70, llama-benchy
0.4.0, total tokens/s:

| Test | Users | v1.7.1 | **v1.8** |
|---|---|---:|---:|
| pp512 / tg128 | 1 / 2 / 4 / 8 | 49.0 / 67.4 / 84.2 / 139.2 | **63.5 / 93.7 / 118.2 / 187.0** |
| pp4096 / tg128 | 1 / 2 / 4 / 8 | 51.1 / 38.4 / 36.2 / 39.7 | **69.6 / 48.0 / 46.0 / 48.1** |
| prompt pp4096 | 1 | 1,731 | **2,078** |

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
  3 drafts 61.9 / 56.0, 4 drafts 63.5 / 69.6, 5 drafts 66.2 / 65.0 tok/s.
- The MTP head is stored in MXFP4. Drafting takes 5.1 instead of 9.2 ms per step. The head
  only proposes tokens: every token you get is still the model's own greedy choice.
  `GRIMOIRE_MTP_HEAD_FMT=bf16` keeps it in BF16.
- Drafting stays on while up to 4 requests are active (`GRIMOIRE_SPEC_MAX_SEQS`, was 2).

**INT4 verify faster.** For 2 to 8 rows (the MTP check and batched decoding), the INT4
matrix kernel now runs with 128 registers per thread and splits the work over more threads:
42.2 -> 38.0 ms per MTP check. The output layer is also checked in one pass instead of row by row.

**INT4 prompt processing.** GPTQ / INT4 weights now go through the fast prompt path that
MXFP4 used, with the feed-forward gate fused in: 1,731 -> 2,078 tok/s at 4,096 tokens, same
output text.

### Comparison on the same card

vLLM 0.30.1 was run with the flags from the
[B70 inference cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) for
this checkpoint, in its own container on the same B70. That means MTP with 4 drafts, fp8 KV
cache and GPTQ, with no cookbook patches and the head in BF16. The same llama-benchy runs gave:

| Test | Users | vLLM 0.30.1 | GRIMOIRE v1.8 |
|---|---|---:|---:|
| pp512 / tg128 | 1 / 2 / 4 / 8 | 45.1 / 85.2 / 130.4 / 122.3 | 63.5 / 93.7 / 118.2 / 187.0 |
| pp4096 / tg128 | 1 / 2 / 4 / 8 | 40.1 / 57.8 / 64.3 / 46.0 | 69.6 / 48.0 / 46.0 / 48.1 |
| prompt pp4096 | 1 | 1,948 | 2,078 |

vLLM's MTP accepted 37% of its drafts on llama-benchy's text. The cookbook's own 84-113 tok/s
come from prompts where about 95% of drafts are accepted.

### Still open

- With several long prompts at once, GRIMOIRE prefills them one after another, and the other
  users wait (pp4096 at 2 and 4 users). Mixing prefill with ongoing decoding is the next item.
- Tool calling in the server.

### Image

`grimoire-b70-v1.8.tar.gz`, SIZE bytes, sha256 `SHA256`, built from commit COMMIT (image id
`IMAGEID`).
