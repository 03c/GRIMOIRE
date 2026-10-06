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

### Unchanged

llama-benchy, Qwen3.8-27B GPTQ + MTP (8 slots, `--ctx 16384`): tg128 67.6 / 187.2 tok/s at
1 / 8 users after pp512, prompt pp4096 2,079 tok/s, coherence test passed. pp512 reads ~9% lower
(1,297 vs ~1,420) only because the server now sends Qwen3.8's real template, with its
reasoning-effort system text, which llama-benchy does not count as prompt.

### Image

`grimoire-b70-v1.8.2.tar.gz`, SIZE bytes, sha256 `SHA256`, built from commit COMMIT (image id
`IMAGEID`).
