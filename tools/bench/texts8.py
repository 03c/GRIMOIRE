#!/usr/bin/env python3
"""texts8.py PORT MODEL [N] -- 8 concurrent greedy chats; prints JSON {i: text} (batched decode check)."""
import json, sys, threading, urllib.request
port, model = int(sys.argv[1]), sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
P = ["Explain in two sentences why the sky is blue.", "Write a haiku about a lighthouse.",
     "List three prime numbers greater than 100.", "What is the capital of Australia? Answer briefly.",
     "Describe a sunset over the ocean in one paragraph.", "Give two tips for learning a new language.",
     "Summarize the plot of Hamlet in three sentences.", "Why do cats purr? Answer in two sentences."]
out = {}
def chat(i):
    body = json.dumps({"model": f"/models/{model}", "messages": [{"role": "user", "content": P[i]}],
                       "max_tokens": N, "temperature": 0}).encode()
    with urllib.request.urlopen(urllib.request.Request(f'http://localhost:{port}/v1/chat/completions',
            data=body, headers={'Content-Type': 'application/json'}), timeout=900) as r:
        d = json.load(r)
    out[i] = d['choices'][0]['message'].get('content') or d['choices'][0]['message'].get('reasoning_content') or ''
ts = [threading.Thread(target=chat, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
print(json.dumps({str(k): out[k] for k in sorted(out)}, indent=1, ensure_ascii=False))
