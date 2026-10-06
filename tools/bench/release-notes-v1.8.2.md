## GRIMOIRE v1.8.2 -- tool calling, reasoning_content, no 2-second stalls when a long prompt arrives

```
docker load < grimoire-b70-v1.8.2.tar.gz      # -> grimoire-b70:latest
```

### New

**OpenAI tool calling.** `tools` / `tool_choice` requests are supported on Qwen3.8-27B (and its
fine-tunes), Ornith-1.5-35B-A3B, Agnes-3.0-Flash and Qwen3.8-Flash-Next. The prompt is rendered
byte-for-byte as each checkpoint's own Jinja chat template. That was checked on 13 request cases
(tools, tool results, multi-turn, thinking on/off, reasoning effort) for each of these models. The
model's `<tool_call>` blocks come back as OpenAI `tool_calls` with `finish_reason: "tool_calls"`,
streamed or not. Qwen3.8-27B GPTQ + MTP, asked "What is the weather in Paris?" with a
`get_weather` tool:

- it calls `get_weather({"city": "Paris", "unit": "celsius"})`;
- given the tool's answer, it replies "The current weather in Paris is 18°C (64°F) with light
  rain ...".

Qwen3.6-35B-A3B's newer template is not supported yet, nor are the K2 and Muse templates. A request
with `tools` gets HTTP 400 on those models.

**Reasoning in `reasoning_content`.** For thinking models (Qwen3.8, Ornith, Agnes, Flash-Next, K2),
the answer is in `content` and the reasoning in `reasoning_content`, as OpenAI clients expect.
`chat_template_kwargs: {"enable_thinking": false}` turns thinking off.

**A long prompt no longer freezes the other users.** A prompt longer than 1,024 tokens that arrives
while other requests are decoding is processed in 1,024-token chunks, with the others' decoding in
between (`GRIMOIRE_INTERLEAVE_CHUNK`, 0 = off). Measured on Qwen3.8-27B GPTQ + MTP: user B sent
a 4K-token prompt while user A was streaming an answer.

| | A's longest pause | B's first token |
|---|---:|---:|
| v1.8.1 | 2.03 s | 1.99 s |
| v1.8.2 | 0.67 s | 2.58 s |

B's answer was identical either way.

### Speed

The server now sends Qwen3.8's official chat template, the same prompt vLLM sends. It adds the
template's reasoning-effort system text, and the model's reasoning, and so how many of MTP's
drafts it accepts, shifts a little.

| | 1 user, tg128 after pp512 (llama-benchy, 4 runs) |
|---|---:|
| v1.8.1 | 63.8 ± 5.1 tok/s |
| v1.8.2 | 58.3 ± 6.5 tok/s |

vLLM 0.30.1 with the same checkpoint and flags measured 45.1. Release validation on the v1.8.2
image (8 slots, `--ctx 16384`, coherence test passed):

| Test | 1 user | 8 users |
|---|---:|---:|
| pp512 / tg128 | 50.5 | 187.4 |
| pp4096 / tg128 | 44.9 | 148.9 |
| prompt pp4096 | 2,020 | |

pp512 reads lower (1,305 vs ~1,420) only because llama-benchy does not count the template's
system text as prompt.

### Image

`grimoire-b70-v1.8.2.tar.gz`, 640,927,385 bytes, sha256
`b0a0e542b697719db53171d04f202bc5655f6d364e1a795c7532f3f2ee3cb1b5`, built from commit 1ac79b4 (image id
`83b3d39a3d01`).
