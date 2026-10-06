#!/usr/bin/env python3
# spec_probe.py URL MODEL [metrics] -- same chat prompts, greedy, 256 tokens; prints tok/s and
# (vLLM) spec-decode acceptance per prompt from /metrics counter deltas.
import json, sys, time, urllib.request, re
url, model = sys.argv[1], sys.argv[2]
use_metrics = len(sys.argv) > 3 and sys.argv[3] == "metrics"
serg = open("/mnt/storage/isos/grimoire-runs/bench-1003/sergio_prompt.txt").read()
prompts = {
  "story": "Write a detailed story about a lighthouse keeper.",
  "code": "Write a Python function that parses a CSV file and returns the average of each numeric column. Include error handling.",
  "serg": serg,
}
def metrics():
    if not use_metrics: return {}
    t = urllib.request.urlopen(url.replace("/v1", "") + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:spec_decode_num_drafts_total"):
        m = [float(x) for x in re.findall(r"^" + re.escape(name) + r"\{[^}]*\} ([0-9.e+]+)$", t, re.M)]
        out[name] = sum(m)
    return out
for name, p in prompts.items():
    before = metrics()
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 256,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; n = 0; text = []; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
            if piece:
                if first is None: first = time.time()
                last = time.time(); n += 1; text.append(piece)
    t1 = time.time()
    after = metrics()
    acc = ""
    if use_metrics:
        a = after["vllm:spec_decode_num_accepted_tokens_total"] - before["vllm:spec_decode_num_accepted_tokens_total"]
        dr = after["vllm:spec_decode_num_draft_tokens_total"] - before["vllm:spec_decode_num_draft_tokens_total"]
        ds = after["vllm:spec_decode_num_drafts_total"] - before["vllm:spec_decode_num_drafts_total"]
        acc = f"accepted {a:.0f}/{dr:.0f} drafted ({100*a/max(dr,1):.1f}%), {a/max(ds,1)+1:.2f} tokens/step"
    full = "".join(text)
    toks = usage.get("completion_tokens", n) if usage else n
    span = (last or t1) - (first or t1)
    print(f"{name:6s} tokens {toks:4d} (chunks {n})  TTFT {(first or t1)-t0:5.2f}s  decode {(toks-1)/max(span,1e-9):6.1f} tok/s  {acc}")
    print("   text:", full[:160].replace("\n", " | "))
